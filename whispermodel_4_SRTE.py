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
import torch
from tool import *


class WhisperNERModel(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # 记录超参更稳, 通过self.hparams.XXX调用
        
        self.whisper = WhisperForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(self.hparams.processor_name_or_path, language="en", task="transcribe")
        
        task_tokens = ["<|task_ASR|>", "<|task_NER|>", "<|task_RE|>", "<|task_RTE|>"]
        tokenizer, num_added = add_special_tokens(self.processor.tokenizer, task_tokens)
        if num_added > 0:
            self.whisper.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer

        # base_forced = self.tokenizer.get_decoder_prompt_ids(language="en", task="transcribe")
        # base_ids = [tid for _, tid in base_forced]

        rte_id_1 = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        rte_id_2 = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        rte_id_3 = self.tokenizer.convert_tokens_to_ids("<|task_RTE|>")
        prefix_ids = [rte_id_1, rte_id_2, rte_id_3]
        self.prefix = torch.tensor(prefix_ids, dtype=torch.long)  # ✅ 1D





        # symbal->Type
        self.symbal2nertype = {
            '<':'ORG-S',
            '>':'ORG-E',
            '(':'LOC-S',
            ')':'LOC-E',
            '[':'PER-S',
            ']':'PER-E'
        }

        # for st in self.whisper.generation_config.suppress_tokens:
        #     print(self.tokenizer.convert_ids_to_tokens(st))
        # exit()

        # 白名单
        self.white_list = ['<', '>', '(', ')', '[', ']', '-', '$', '$$', '#', '##', ]


        # print(self.whisper.generation_config.suppress_tokens)
        for wl in self.white_list:
            if self.tokenizer.convert_tokens_to_ids(wl) in self.whisper.generation_config.suppress_tokens:
                # print(wl)
                self.whisper.generation_config.suppress_tokens.remove(self.tokenizer.convert_tokens_to_ids(wl))
        # print(self.whisper.generation_config.suppress_tokens)

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

        # labels = batch["labels"].clone()
        # labels[labels == -100] = self.tokenizer.pad_token_id
        # lab_text_batch = self.processor.batch_decode(labels, skip_special_tokens=False)
        # print(lab_text_batch)
        # exit()

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

        self.whisper.generation_config.forced_decoder_ids = None
        self.whisper.generation_config.return_timestamps = False

        gen_outputs = self.whisper.generate(
            input_features=batch["input_features"].to(self.whisper.device),
            # attention_mask=batch["attention_mask"].to(self.whisper.device),
            prompt_ids=self.prefix.to(self.whisper.device),   # ✅ 用 prompt_ids
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
            return_timestamps=False,
        )


        # gen_outputs = self.whisper.generate(
        #     input_features=batch["input_features"],
        #     attention_mask=batch["attention_mask"],
        #     # forced_decoder_ids=forced_decoder_ids,
        #     language="en",
        #     task="transcribe",
        #     return_timestamps=False,
        #     decoder_input_ids=self.prefix.to(self.device),
        #     max_new_tokens=128,
        # )
        labels = batch["labels"].clone()
        labels[labels == -100] = self.tokenizer.pad_token_id
        gen_text_batch = self.processor.batch_decode(gen_outputs, skip_special_tokens=True)
        lab_text_batch = self.processor.batch_decode(labels, skip_special_tokens=True)
        # print(gen_text_batch)
        # print("-------------------------------------------------------------------------")
        # print(lab_text_batch)
        # exit()


        batch_rte_lab = self.batch_text2rte(lab_text_batch)
        batch_rte_gen = self.batch_text2rte(gen_text_batch)
        self.compute_metric_step_update_4_rte(batch_rte_lab, batch_rte_gen)

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

    def text2rte(self, text):
        # text_re = "Douglas Flint##person title##chairman$$Douglas Flint##person title##Chief Financial Officer"
        final_rte = []
        for re_item in text.split("$$"):
            re_item = re_item.strip()
            if re_item:
                final_rte.append(re_item)
        return final_rte


    def batch_text2rte(self, batch_text):
        batch_rte = []
        for text_item in batch_text:
            batch_rte.append(self.text2rte(text_item))
        return batch_rte

    def compute_metric_step_update_4_rte(self, batch_rte_label, batch_rte_pred):
        for bel, bep in zip(batch_rte_label, batch_rte_pred):
            # 更新span无位置的PRC
            # batch_entities_label_without_index = ['中国-LOC']
            batch_rte_label = [l for l in bel if l != "None"]
            batch_rte_pred = [p for p in bep if p != "None"]
            self.P_E += len(set(batch_rte_pred))
            self.R_E += len(set(batch_rte_label))
            self.C_E += len(set(batch_rte_pred) & set(batch_rte_label))

    def clear_PRC(self):
        self.P_E = 0.0
        self.R_E = 0.0
        self.C_E = 0.0
        self.P_E_S = 0.0
        self.R_E_S = 0.0
        self.C_E_S = 0.0



