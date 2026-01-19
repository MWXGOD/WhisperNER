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

        # # 定义PRF的变量
        # self.P_E = 0.0
        # self.R_E = 0.0
        # self.C_E = 0.0
        # self.P_E_S = 0.0
        # self.R_E_S = 0.0
        # self.C_E_S = 0.0

        # 定义PRF的变量
        self.P_NER = 0.0
        self.R_NER = 0.0
        self.C_NER = 0.0
        self.P_RE = 0.0
        self.R_RE = 0.0
        self.C_RE = 0.0
        self.P_RTE = 0.0
        self.R_RTE = 0.0
        self.C_RTE = 0.0

        # 定义f1
        self.max_f1 = 0.0


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
        val_loss = None
        if accelerator is not None:
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

        labels = batch["labels"].clone()
        relations = batch["relations"]
        entities = batch["entities"]
        labels[labels == -100] = self.tokenizer.pad_token_id
        gen_text_batch = self.processor.batch_decode(gen_outputs, skip_special_tokens=True)
        lab_text_batch = self.processor.batch_decode(labels, skip_special_tokens=True)

        batch_rte_lab = self.batch_text2rte(lab_text_batch)
        batch_rte_gen = self.batch_text2rte(gen_text_batch)
        self.compute_metric_step_update_4_rte(batch_rte_lab, batch_rte_gen)
        self.compute_metric_step_update_4_re(relations, batch_rte_gen)
        self.compute_metric_step_update_4_ner(entities, batch_rte_gen)

        return gen_text_batch, lab_text_batch, val_loss

    def on_validation_epoch_end(self):
        P_NER = self.C_NER / self.P_NER if self.P_NER > 0.0 else 0.0
        R_NER = self.C_NER / self.R_NER if self.R_NER > 0.0 else 0.0
        F1_NER = 2 * P_NER * R_NER / (P_NER + R_NER) if (P_NER + R_NER) > 0.0 else 0.0

        P_RE = self.C_RE / self.P_RE if self.P_RE > 0.0 else 0.0
        R_RE = self.C_RE / self.R_RE if self.R_RE > 0.0 else 0.0
        F1_RE = 2 * P_RE * R_RE / (P_RE + R_RE) if (P_RE + R_RE) > 0.0 else 0.0

        P_RTE = self.C_RTE / self.P_RTE if self.P_RTE > 0.0 else 0.0
        R_RTE = self.C_RTE / self.R_RTE if self.R_RTE > 0.0 else 0.0
        F1_RTE = 2 * P_RTE * R_RTE / (P_RTE + R_RTE) if (P_RTE + R_RTE) > 0.0 else 0.0
        if self.max_f1 < F1_RTE:
            self.max_f1 = F1_RTE

        return P_NER, R_NER, F1_NER, P_RE, R_RE, F1_RE, P_RTE, R_RTE, F1_RTE

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

    def compute_metric_step_update_4_rte(self, batch_rte_lab, batch_rte_gen):
        for brl, brp in zip(batch_rte_lab, batch_rte_gen):
            brl = [l for l in brl if l != "None"]
            brp = [p for p in brp if p != "None"]
            self.P_RTE += len(set(brp))
            self.R_RTE += len(set(brl))
            self.C_RTE += len(set(brp) & set(brl))

    def compute_metric_step_update_4_re(self, relations, batch_rte_gen):
        for rel, brp in zip(relations, batch_rte_gen):
            relations_labels = rel
            relations_preds = []
            for p in brp:
                if len(p.split("##")) == 3:
                    relations_preds.append(p.split("##")[1])
            self.P_RE += len(set(relations_preds))
            self.R_RE += len(set(relations_labels))
            self.C_RE += len(set(relations_preds) & set(relations_labels))

    def compute_metric_step_update_4_ner(self, entities, batch_rte_gen):
        for ent, brp in zip(entities, batch_rte_gen):
            entities_labels = ent
            entities_preds = []
            for p in brp:
                if len(p.split("##")) == 3:
                    entities_preds.append(p.split("##")[0])
                    entities_preds.append(p.split("##")[2])
            self.P_NER += len(set(entities_preds))
            self.R_NER += len(set(entities_labels))
            self.C_NER += len(set(entities_preds) & set(entities_labels))

    def clear_PRC(self):
        self.P_NER = 0.0
        self.R_NER = 0.0
        self.C_NER = 0.0
        self.P_RE = 0.0
        self.R_RE = 0.0
        self.C_RE = 0.0
        self.P_RTE = 0.0
        self.R_RTE = 0.0
        self.C_RTE = 0.0



