import os, json, time, torch, argparse, swanlab
from tool import Hyperargs, read_json_args
from tqdm import tqdm
from datetime import datetime
from lightning.pytorch import seed_everything
from whispermodel_4_SRTE import WhisperNERModel
from loguru import logger
from accelerate import Accelerator
from torch.optim import AdamW
from data_module_4_SRTE import WhisperNERDataModule
from transformers import get_constant_schedule_with_warmup


parser = argparse.ArgumentParser()
parser.add_argument('--args_path', type=str, default='argsfile/SRTE/ReTACRED_small.json')
# parser.add_argument('--args_path', type=str, default='argsfile/aishell_ner_args_4_whisper_medium.json')
parser.add_argument('--decode_schema', type=str, default=None)
shell_args = parser.parse_args()
args_dict = read_json_args(shell_args.args_path)
hyperargs = Hyperargs(**args_dict)
hyperargs.decode_schema = shell_args.decode_schema
seed_everything(hyperargs.seed, workers=True)

time_str = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
swanlab.init(
    project=hyperargs.swanlab_project_name,
    config=hyperargs.__dict__,
    experiment_name=f"MWX-Whisper-Demo-{time_str}"
)
# 数据
data_module = WhisperNERDataModule(**hyperargs.__dict__)
data_module.setup(stage = "test")
test_dataloader = data_module.test_dataloader()

# 模型
# 读取模型
save_path = os.path.join(hyperargs.output_model_path, f"{hyperargs.output_model_path.split('/')[-1]}.bin")
ckpt = torch.load(save_path, map_location="cpu")
model = WhisperNERModel(**ckpt["hparams"])
model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval()
model.to(hyperargs.gpu_id)

# 推理
with torch.no_grad():
    test_start = time.time()
    model.eval()
    model.clear_PRC()
    test_bar = tqdm(
        test_dataloader,
        desc=f"Testing",
        leave=False
    )
    gen_text_per_epoch = []
    lab_text_per_epoch = []
    for batch in test_bar:
        gen_text_batch, lab_text_batch, _ = model.validation_step(batch)
        gen_text_per_epoch += gen_text_batch
        lab_text_per_epoch += lab_text_batch
    P_NER, R_NER, F1_NER, P_RE, R_RE, F1_RE, P_RTE, R_RTE, F1_RTE = model.on_validation_epoch_end()
    swanlab.log({"test_F1_NER": F1_NER, "test_P_NER": P_NER, "test_R_NER": R_NER})
    swanlab.log({"test_F1_RE": F1_RE, "test_P_RE": P_RE, "test_R_RE": R_RE})
    swanlab.log({"test_F1_RTE": F1_RTE, "test_P_RTE": P_RTE, "test_R_RTE": R_RTE})
    print(f"test_F1_NER: {F1_NER}, test_P_NER: {P_NER}, test_R_NER: {R_NER}, test_F1_RE: {F1_RE}, test_P_RE: {P_RE}, test_R_RE: {R_RE}, test_F1_RTE: {F1_RTE}, test_P_RTE: {P_RTE}, test_R_RTE: {R_RTE}")
    test_end = time.time()
    logger.info(f"测试结束，总时长：{(test_end-test_start)/60:.2f}分钟")
    logger.info(f"测试的三元组F1：{F1_RTE}")







