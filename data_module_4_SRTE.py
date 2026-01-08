import json, os
from typing import Any
from transformers import AutoProcessor, WhisperForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import lightning as L
import librosa
import torch
from torch.utils.data import Subset



class WhisperNERDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.read_data()

    def read_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]
        return data_item["audio_path"], data_item["target_text"]



class WhisperNERDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path,
        processor_name_or_path,
        batch_size=8,
        num_workers=4,
        max_length=128,
        sample_rate=16000,
        **kwargs
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path, language="en", task="transcribe")
        self.save_hyperparameters()  # 记录超参更稳, 通过self.hparams.XXX调用

    def setup(self, stage=None):
        if stage in (None, "fit"):
            # full_train = WhisperNERDataset(
            #     os.path.join(self.data_path, "train_target.json")
            # )

            # self.train_dataset = Subset(full_train, range(32))
            self.train_dataset = WhisperNERDataset(os.path.join(self.data_path, "train_target.json"))
            self.dev_dataset = WhisperNERDataset(os.path.join(self.data_path, "dev_target.json"))
        if stage in (None, "test"):
            self.test_dataset  = WhisperNERDataset(os.path.join(self.data_path, "test_target.json"))

    def collate_fn(self, batch):

        audio_list = [] # array (numpy)
        texts_list = []

        for x in batch:
            audio_path, text = x
            wav, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)  # 直接加载为指定采样率和单声道
            audio_list.append(wav)
            texts_list.append(text)

        audio_features = self.processor(
            audio_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            # padding="longest",              # 或 "max_length" 不能padding longest，源码里写死了，必须3000.
            # padding="max_length",  # 使用max_length填充
            # max_length=3000,       # 确保长度为3000
            return_attention_mask=True,
        )
        # print(self.hparams.decode_schema)
        task_tokens = ["<|task_ASR|>", "<|task_NER|>", "<|task_RE|>", "<|task_RTE|>"]
        num_added = self.processor.tokenizer.add_special_tokens({
            "additional_special_tokens": task_tokens
        })

        if self.hparams.decode_schema == "E-T8558":
            label_features = self.processor.tokenizer(
                entities_types_list,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
        else:
            # 直接使用tokenizer处理目标文本
            label_features = self.processor.tokenizer(
                texts_list,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )

        labels = label_features["input_ids"]
        labels_attention_mask = label_features["attention_mask"]
        labels = labels.masked_fill(labels_attention_mask == 0, -100)

        return {
            "input_features": audio_features["input_features"],
            "attention_mask": audio_features["attention_mask"],
            "labels": labels,
        }


    def train_dataloader(self):
        return DataLoader[Any](
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def dev_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )


if __name__ == "__main__":
    
    from transformers import WhisperTokenizer

    tokenizer = WhisperTokenizer.from_pretrained(
        "cache/whisper-small",
        language="en",
        task="transcribe"
    )

    task_tokens = ["<|task_ASR|>", "<|task_NER|>", "<|task_RE|>", "<|task_RTE|>"]

    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": task_tokens
    })

    print("Added:", num_added)
    print("RTE id:", tokenizer.convert_tokens_to_ids("<|task_RTE|>"))
    print(tokenizer.tokenize("<|task_RTE|>"))



    # data_path = "data/aishell"
    # processor_name_or_path = "cache/whisper-small"
    # batch_size = 8
    # num_workers = 4
    # max_length = 128
    # sample_rate = 16000
    # # processor = AutoProcessor.from_pretrained(processor_name_or_path, language="zh")
    # model = WhisperForConditionalGeneration.from_pretrained(processor_name_or_path)
    # kwargs = {
    #     "decode_schema": "E-T8558",
    # }


    # dm = WhisperNERDataModule(
    #     data_path,
    #     processor_name_or_path,
    #     batch_size,
    #     num_workers,
    #     max_length,
    #     sample_rate,
    #     **kwargs,
    # )
    # dm.setup(stage="fit")
    # train_loader = dm.train_dataloader()
    # for batch in train_loader:
    #     print(batch["input_features"].shape)
    #     print(batch["labels"].shape)
    #     generated_ids = model.generate(input_features=batch["input_features"])
    #     transcription = dm.processor.batch_decode(generated_ids, skip_special_tokens=True)

    #     labels = batch["labels"].clone()
    #     labels[labels == -100] = dm.processor.tokenizer.pad_token_id
    #     labels = dm.processor.batch_decode(labels, skip_special_tokens=True)
    #     print(transcription)
    #     print(labels)   
    #     break





