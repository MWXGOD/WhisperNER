import json, os
from typing import Any
from transformers import AutoProcessor, WhisperForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import lightning as L
import librosa
import torch



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
        audio_path = os.path.join(
            os.path.dirname(self.data_path),
            "audio",
            f"{data_item['origin_id']}.wav"
        )
        return audio_path, data_item["transcript"]



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
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path, language="zh")

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = WhisperNERDataset(os.path.join(self.data_path, "train.json"))
            self.dev_dataset = WhisperNERDataset(os.path.join(self.data_path, "dev.json"))
        if stage in (None, "test"):
            self.test_dataset  = WhisperNERDataset(os.path.join(self.data_path, "test.json"))

    def collate_fn(self, batch):

        audio_list = [] # array (numpy)
        texts_list = []

        for x in batch:
            audio_path, text = x
            wav, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)  # 直接加载为指定采样率和单声道
            audio_list.append(wav)
            texts_list.append(text)

        inputs = self.processor(
            audio_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            # padding="max_length",  # 使用max_length填充
            # max_length=3000,       # 确保长度为3000
            truncation=True,       # 截断过长的序列
        )

        # 直接使用tokenizer处理目标文本
        label_features = self.processor.tokenizer(
            texts_list,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        labels = label_features["input_ids"]
        attn = label_features["attention_mask"]
        labels = labels.masked_fill(attn == 0, -100)

        return {
            "input_features": inputs["input_features"],
            # "attention_mask": inputs["input_features"],
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
    data_path = "data/aishell"
    processor_name_or_path = "cache/whisper-small"
    batch_size = 8
    num_workers = 4
    max_length = 128
    sample_rate = 16000
    # processor = AutoProcessor.from_pretrained(processor_name_or_path, language="zh")
    model = WhisperForConditionalGeneration.from_pretrained(processor_name_or_path)


    dm = WhisperNERDataModule(
        data_path,
        processor_name_or_path,
        batch_size,
        num_workers,
        max_length,
        sample_rate,
    )
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        print(batch["input_features"].shape)
        print(batch["labels"].shape)
        generated_ids = model.generate(input_features=batch["input_features"], language="zh")
        transcription = dm.processor.batch_decode(generated_ids, skip_special_tokens=True)

        labels = batch["labels"].clone()
        labels[labels == -100] = dm.processor.tokenizer.pad_token_id
        labels = dm.processor.batch_decode(labels, skip_special_tokens=True)
        print(transcription)
        print(labels)   
        break





