import lightning as L
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_constant_schedule_with_warmup
from transformers import AutoProcessor, WhisperForConditionalGeneration
from lightning.pytorch.loggers import WandbLogger
import swanlab
from data_module import WhisperNERDataModule
from torch.optim import AdamW
from datetime import datetime
from torch.nn.utils import clip_grad_norm_ as clip
from tqdm import tqdm



class WhisperNERModel(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # 记录超参更稳, 通过self.hparams.XXX调用
        
        self.whisper = WhisperForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(self.hparams.processor_name_or_path)
        self.tokenizer = self.processor.tokenizer

        # symbal->Type
        self.symbal2nertype = {
            '<':'ORG-S',
            '>':'ORG-E',
            '(':'LOC-S',
            ')':'LOC-E',
            '[':'PER-S',
            ']':'PER-E'
        }

        # 白名单
        self.white_list = ['<', '>', '(', ')', '[', ']']
        for wl in self.white_list:
            self.whisper.generation_config.suppress_tokens.remove(self.tokenizer.convert_tokens_to_ids(wl))

        # 定义PRF的变量
        self.P_E = 0.0
        self.R_E = 0.0
        self.C_E = 0.0
        self.P_E_S = 0.0
        self.R_E_S = 0.0
        self.C_E_S = 0.0

        # 定义f1
        self.max_f1 = 0.0
        self.max_f1_S = 0.0


    def forward(self, input_features, labels=None):
        outputs = self.whisper(
            input_features=input_features,
            labels=labels,
        )
        return outputs

    def on_train_batch_start(self, epoch, train_dataloader):
        self.train()
        train_bar = tqdm(
            train_dataloader,
            desc=f"Epoch [{epoch+1}/{self.hparams.epochs_num}] Training",
            leave=False
        )
        return train_bar

    def training_step(self, batch, optimizer=None, scheduler=None, accelerator=None):
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            outputs = self(input_features=batch["input_features"], labels=batch["labels"])
            train_loss = outputs.loss
        accelerator.backward(train_loss)
        accelerator.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        return train_loss

    def on_validation_batch_start(self, epoch, dev_dataloader):
        self.eval()
        self.clear_PRC()
        dev_bar = tqdm(
            dev_dataloader,
            desc=f"Epoch [{epoch+1}/{self.hparams.epochs_num}] Validation",
            leave=False
        )
        return dev_bar

    def validation_step(self, batch, optimizer=None, scheduler=None, accelerator=None):
        with accelerator.autocast():
            outputs = self(input_features=batch["input_features"], labels=batch["labels"])
            val_loss = outputs.loss
        # forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="zh")
        gen_outputs = self.whisper.generate(
            input_features=batch["input_features"],
            # attention_mask=batch["attention_mask"],
            # forced_decoder_ids=forced_decoder_ids,
            # language="zh",
            max_new_tokens=128,
        )
        labels = batch["labels"].clone()

        
        # gen_text_batch = self.processor.batch_decode(gen_outputs, skip_special_tokens=False)
        # lab_text_batch = self.processor.batch_decode(labels, skip_special_tokens=False)
        # print(gen_text_batch[0])
        # print(lab_text_batch[0])
        # exit()




        labels[labels == -100] = self.tokenizer.pad_token_id
        gen_text_batch = self.processor.batch_decode(gen_outputs, skip_special_tokens=True)
        lab_text_batch = self.processor.batch_decode(labels, skip_special_tokens=True)

        batch_entities_lab = self.batch_text2entity(lab_text_batch, is_labels = True)
        batch_entities_gen = self.batch_text2entity(gen_text_batch)
        self.compute_metric_step_update(batch_entities_lab, batch_entities_gen)

        return gen_text_batch, lab_text_batch, val_loss

    def on_validation_epoch_end(self):
        P = self.C_E / self.P_E if self.P_E > 0.0 else 0.0
        R = self.C_E / self.R_E if self.R_E > 0.0 else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) > 0.0 else 0.0

        P_S = self.C_E_S / self.P_E_S if self.P_E_S > 0.0 else 0.0
        R_S = self.C_E_S / self.R_E_S if self.R_E_S > 0.0 else 0.0
        F1_S = 2 * P_S * R_S / (P_S + R_S) if (P_S + R_S) > 0.0 else 0.0
        if F1 > self.max_f1:
            self.max_f1 = F1
        if F1_S > self.max_f1_S:
            self.max_f1_S = F1_S
        # swanlab.log({"F1": F1, "F1_S": F1_S})
        return P, R, F1, P_S, R_S, F1_S

    def text2entity(self, text):
        # text = "<胡润>(中国)私人财富管理白皮书调查显示"
        # final = [['中国', 'LOC', [2, 4]], ['胡润', 'ORG', [0, 2]]]
        stack = []
        reversed_stack = []
        entities = []
        reversed_entities = []
        shift_index = [] # 避免端点冲突
        reversed_shift_index = [] # 避免端点冲突
        for i, char in enumerate(text):
            if char in self.symbal2nertype:
                shift_index.append(shift_index[-1] + 1  if shift_index else 1)
                if stack:
                    push_flag = 1
                    for j in reversed(range(len(stack))):
                        prior_type = self.symbal2nertype[stack[j][1]]
                        curre_type = self.symbal2nertype[char]
                        if prior_type[0:-1] == curre_type[0:-1] and prior_type[-1] == 'S' and curre_type[-1] == 'E':
                            entities.append([text[stack[j][0]+1: i], curre_type[0:-2], [stack[j][0]+1-shift_index[stack[j][0]], i-shift_index[i-1]]])
                            stack.remove(stack[j])
                            push_flag = 0
                            break
                    if push_flag == 1:
                        stack.append((i, char))
                else:
                    stack.append((i, char))
            else:
                shift_index.append(shift_index[-1] if shift_index else 0)

        # 反向解码
        text = ''.join(reversed(text))
        for i, char in enumerate(text):
            if char in self.symbal2nertype:
                reversed_shift_index.append(reversed_shift_index[-1] + 1  if reversed_shift_index else 1)
                if reversed_stack:
                    push_flag = 1
                    for j in reversed(range(len(reversed_stack))):
                        prior_type = self.symbal2nertype[reversed_stack[j][1]]
                        curre_type = self.symbal2nertype[char]
                        if prior_type[0:-1] == curre_type[0:-1] and prior_type[-1] == 'E' and curre_type[-1] == 'S':
                            reversed_entities.append([(text[reversed_stack[j][0]+1: i]), curre_type[0:-2], [reversed_stack[j][0]+1-reversed_shift_index[reversed_stack[j][0]], i-reversed_shift_index[i-1]]])
                            reversed_stack.remove(reversed_stack[j])
                            push_flag = 0
                            break
                    if push_flag == 1:
                        reversed_stack.append((i, char))
                else:
                    reversed_stack.append((i, char))
            else:
                reversed_shift_index.append(reversed_shift_index[-1] if reversed_shift_index else 0)
        for e in reversed_entities:
            e[0] = ''.join(reversed(e[0]))
            temp = e[2][0]
            e[2][0] = len(text) - reversed_shift_index[-1] - e[2][1]
            e[2][1] = len(text) - reversed_shift_index[-1] - temp

        final_entities = []
        for e in entities:
            for re in reversed_entities:
                if e == re:
                    final_entities.append(e)
        return final_entities

    def batch_text2entity(self, batch_text, is_labels=False):
        batch_entities = []
        text_item_entities_temp = []
        if is_labels:
            for text_item in batch_text:
                batch_entities.append(self.text2entity(text_item))
        else:
            for text_id, text_item in enumerate(batch_text):
                text_item_entities = self.text2entity(text_item)
                text_item_entities_temp += text_item_entities
                if (text_id+1)%self.hparams.num_return_sequences == 0:
                    text_item_entities_vote = []
                    for entity in text_item_entities_temp:
                        if text_item_entities_temp.count(entity) > self.hparams.num_return_sequences/2 and entity not in text_item_entities_vote:
                            text_item_entities_vote.append(entity)
                    batch_entities.append(text_item_entities_vote)
                    text_item_entities_temp = []
        return batch_entities

    def compute_metric_step_update(self, batch_entities_label, batch_entities_pred):

        for bel, bep in zip(batch_entities_label, batch_entities_pred):
            # 更新span位置的P,R,C
            self.P_E_S += len(bep)
            self.R_E_S += len(bel)
            for bep_item in bep:
                if bep_item in bel:
                    self.C_E_S += 1
            # 更新span无位置的PRC
            # batch_entities_label_without_index = ['中国-LOC']
            batch_entities_label_without_index = [f'{l[0]}-{l[1]}' for l in bel]
            batch_entities_pred_without_index = [f'{p[0]}-{p[1]}' for p in bep]
            self.P_E += len(set(batch_entities_pred_without_index))
            self.R_E += len(set(batch_entities_label_without_index))
            self.C_E += len(set(batch_entities_pred_without_index) & set(batch_entities_label_without_index))

    def clear_PRC(self):
        self.P_E = 0.0
        self.R_E = 0.0
        self.C_E = 0.0
        self.P_E_S = 0.0
        self.R_E_S = 0.0
        self.C_E_S = 0.0



