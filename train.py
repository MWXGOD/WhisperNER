import os, json, time, torch, argparse, swanlab
from tool import Hyperargs, read_json_args
from tqdm import tqdm
from datetime import datetime
from lightning.pytorch import seed_everything
from whispermodel import WhisperNERModel
from loguru import logger
from accelerate import Accelerator
from torch.optim import AdamW
from data_module import WhisperNERDataModule
from transformers import get_constant_schedule_with_warmup

# 验证的想法：
# 1. 直接赋值-100，所有endoftext都被覆盖，是否导致模型停不下来？ 已经确认，就是这个问题。
# 2. 学习率是不是导致学得慢？


parser = argparse.ArgumentParser()
parser.add_argument('--args_path', type=str, default='argsfile/aishell_ner_args_4_whisper.json')
shell_args = parser.parse_args()
args_dict = read_json_args(shell_args.args_path)
hyperargs = Hyperargs(**args_dict)
seed_everything(hyperargs.seed, workers=True)

time_str = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
swanlab.init(
    project=hyperargs.swanlab_project_name,
    config=hyperargs.__dict__,
    experiment_name=f"MWX-Whisper-Demo-{time_str}"
)
# 数据
data_module = WhisperNERDataModule(**hyperargs.__dict__)
data_module.setup(stage = "fit")
train_dataloader = data_module.train_dataloader()
dev_dataloader = data_module.dev_dataloader()

# 模型
model = WhisperNERModel(**hyperargs.__dict__)

# 优化器
optimizer = AdamW(model.parameters(), lr=hyperargs.learning_rate, weight_decay = hyperargs.weight_decay)

# 学习率调度器
num_warmup_steps = max(1, int(hyperargs.warmup_rate * hyperargs.epochs_num * len(train_dataloader)))
scheduler = get_constant_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = num_warmup_steps)

# 混合精度
accelerator = Accelerator(mixed_precision=hyperargs.mixed_precision)
model, optimizer, train_dataloader, dev_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, dev_dataloader, scheduler
)

max_f1 = 0
for epoch in range(hyperargs.epochs_num):
    train_start = time.time()
    train_bar = model.on_train_batch_start(epoch, train_dataloader)
    train_loss_per_epoch = 0
    for batch in train_bar:
        loss = model.training_step(batch, optimizer, scheduler, accelerator)
        train_loss_per_epoch += loss.item()
        train_bar.set_postfix({"loss": loss.item()})
    swanlab.log({"train_loss_per_step": train_loss_per_epoch/len(train_dataloader)})
    train_end = time.time()
    logger.info(f"训练结束，第{epoch+1}轮总时长：{(train_end-train_start)/60:.2f}分钟")

    dev_start = time.time()
    dev_bar = model.on_validation_batch_start(epoch, dev_dataloader)
    dev_loss_per_epoch = 0
    gen_text_per_epoch = []
    lab_text_per_epoch = []
    with torch.no_grad():
        for batch in dev_bar:
            gen_text_batch, lab_text_batch, val_loss = model.validation_step(batch, accelerator=accelerator)
            dev_loss_per_epoch += val_loss.item()
            dev_bar.set_postfix({"loss": val_loss.item()})
            gen_text_per_epoch += gen_text_batch
            lab_text_per_epoch += lab_text_batch
    swanlab.log({"dev_loss_per_step": dev_loss_per_epoch/len(dev_dataloader)})
    P, R, F1, P_S, R_S, F1_S = model.on_validation_epoch_end()
    # swanlab.log({"F1": F1, "F1_S": F1_S})
    final_f1 = max(F1, F1_S)
    swanlab.log({"final_f1": final_f1})
    swanlab.log({"P": P, "R": R, "F1": F1, "P_S": P_S, "R_S": R_S, "F1_S": F1_S})
    
    os.makedirs(hyperargs.output_result_path, exist_ok=True)
    with open(f"{hyperargs.output_result_path}/gen_text_batch_{epoch}.json", 'w', encoding='utf-8') as f:
        json.dump({"pred_label": gen_text_per_epoch}, f, indent=4, ensure_ascii=False)
    with open(f"{hyperargs.output_result_path}/lab_text_batch_{epoch}.json", 'w', encoding='utf-8') as f:
        json.dump({"gold_label": lab_text_per_epoch}, f, indent=4, ensure_ascii=False)
    
    if max_f1<final_f1:
        max_f1 = final_f1
        os.makedirs(hyperargs.output_model_path, exist_ok=True)
        save_path = os.path.join(hyperargs.output_model_path, f"{hyperargs.output_model_path.split('/')[-1]}.bin")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hparams": dict(model.hparams),
            }, 
            save_path
        )
        os.makedirs(hyperargs.output_result_path, exist_ok=True)
        with open(f"{hyperargs.output_result_path}/best_gen_text_batch.json", 'w', encoding='utf-8') as f:
            json.dump({"pred_label": gen_text_per_epoch}, f, indent=4, ensure_ascii=False)
        with open(f"{hyperargs.output_result_path}/best_lab_text_batch.json", 'w', encoding='utf-8') as f:
            json.dump({"gold_label": lab_text_per_epoch}, f, indent=4, ensure_ascii=False)
        logger.info("模型已保存")
    logger.info(f"评价指标F: {final_f1:.2f}")
    dev_end = time.time()
    logger.info(f"验证结束，第{epoch+1}轮总时长：{(dev_end-dev_start)/60:.2f}分钟")











