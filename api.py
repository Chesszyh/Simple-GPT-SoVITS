import argparse
import logging
import os
import re
import signal
import subprocess
import sys
from io import BytesIO
from time import time as ttime

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForMaskedLM, AutoTokenizer

import config as global_config
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from feature_extractor import cnhubert
from module.mel_processing import mel_spectrogram_torch, spectrogram_torch
from module.models import SynthesizerTrn, SynthesizerTrnV3

from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from text.LangSegmenter import LangSegmenter
# NOTE 以下方法移到了tools/api_tools.py
from tools.api_tools import DefaultRefer, is_empty, is_full, resample
from tools.my_utils import load_audio

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

# --------------------------------
# 初始化部分
# --------------------------------
dict_language = {
    "中文": "all_zh",
    "粤语": "all_yue",
    "英文": "en",
    "日文": "all_ja",
    "韩文": "all_ko",
    "中英混合": "zh",
    "粤英混合": "yue",
    "日英混合": "ja",
    "韩英混合": "ko",
    "多语种混合": "auto",    #多语种启动切分识别语种
    "多语种混合(粤语)": "auto_yue",
    "all_zh": "all_zh",
    "all_yue": "all_yue",
    "en": "en",
    "all_ja": "all_ja",
    "all_ko": "all_ko",
    "zh": "zh",
    "yue": "yue",
    "ja": "ja",
    "ko": "ko",
    "auto": "auto",
    "auto_yue": "auto_yue",
}

# logger
logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
logger = logging.getLogger('uvicorn')

# 获取配置
g_config = global_config.Config()

# 获取参数
parser = argparse.ArgumentParser(description="GPT-SoVITS api")

# 命令行主要需要调整的参数
parser.add_argument("-s", "--sovits_path", type=str, default=g_config.sovits_path, help="SoVITS模型路径")
parser.add_argument("-g", "--gpt_path", type=str, default=g_config.gpt_path, help="GPT模型路径")

# 基础模型路径
parser.add_argument("-hb", "--hubert_path", type=str, default=g_config.cnhubert_path, help="覆盖config.cnhubert_path")
parser.add_argument("-b", "--bert_path", type=str, default=g_config.bert_path, help="覆盖config.bert_path")

parser.add_argument("-dr", "--default_refer_path", type=str, default=g_config.ref_audio_path , help="默认参考音频路径")
parser.add_argument("-dt", "--default_refer_text", type=str, default=g_config.ref_audio_text, help="默认参考音频文本")
parser.add_argument("-dl", "--default_refer_language", type=str, default=g_config.ref_audio_text_lang, help="默认参考音频语种")

# 设备、端口、音频等配置
parser.add_argument("-d", "--device", type=str, default=g_config.infer_device, help="cuda / cpu")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 0.0.0.0")
parser.add_argument("-p", "--port", type=int, default=g_config.api_port, help="default: 9880")

# bool值传参方法：`python ./api.py -fp ...`，此时 full_precision==True, half_precision==False
parser.add_argument("-fp", "--full_precision", action="store_true", default=False, help="覆盖config.is_half为False, 使用全精度")
parser.add_argument("-hp", "--half_precision", action="store_true", default=False, help="覆盖config.is_half为True, 使用半精度")

parser.add_argument("-sm", "--stream_mode", type=str, default="close", help="流式返回模式, close / normal / keepalive")
parser.add_argument("-mt", "--media_type", type=str, default="wav", help="音频编码格式, wav / ogg / aac")
parser.add_argument("-st", "--sub_type", type=str, default="int16", help="音频数据类型, int16 / int32")

# 切割常用分句符为 `python ./api.py -cp ".?!。？！"`
parser.add_argument("-cp", "--cut_punc", type=str, default="", help="文本切分符号设定, 符号范围,.;?!、，。？！；：…")

args = parser.parse_args()
sovits_path = args.sovits_path
gpt_path = args.gpt_path
device = args.device
port = args.port
host = args.bind_addr
cnhubert_base_path = args.hubert_path
bert_path = args.bert_path
default_cut_punc = args.cut_punc

# 应用参数配置
default_refer = DefaultRefer(args.default_refer_path, args.default_refer_text, args.default_refer_language)

# 模型路径检查
if sovits_path == "":
    sovits_path = g_config.pretrained_sovits_path
    logger.warn(f"未指定SoVITS模型路径, fallback后当前值: {sovits_path}")
if gpt_path == "":
    gpt_path = g_config.pretrained_gpt_path
    logger.warn(f"未指定GPT模型路径, fallback后当前值: {gpt_path}")

# 指定默认参考音频, 调用方 未提供/未给全 参考音频参数时使用
if default_refer.path == "" or default_refer.text == "" or default_refer.language == "":
    default_refer.path, default_refer.text, default_refer.language = "", "", ""
    logger.info("未指定默认参考音频")
else:
    logger.info(f"默认参考音频路径: {default_refer.path}")
    logger.info(f"默认参考音频文本: {default_refer.text}")
    logger.info(f"默认参考音频语种: {default_refer.language}")

# 获取半精度
global is_half = g_config.is_half
if args.full_precision:
    is_half = False
if args.half_precision:
    is_half = True
if args.full_precision and args.half_precision:
    is_half = g_config.is_half  # 炒饭fallback NOTE ?
logger.info(f"半精: {is_half}")

# 流式返回模式
if args.stream_mode.lower() in ["normal","n"]:
    stream_mode = "normal"
    logger.info("流式返回已开启")
else:
    stream_mode = "close"

# 音频编码格式
if args.media_type.lower() in ["aac","ogg"]:
    media_type = args.media_type.lower()
elif stream_mode == "close":
    media_type = "wav"
else:
    media_type = "ogg"
logger.info(f"编码格式: {media_type}")

# 音频数据类型
if args.sub_type.lower() == 'int32':
    is_int32 = True
    logger.info(f"数据类型: int32")
else:
    is_int32 = False
    logger.info(f"数据类型: int16")

# 初始化模型
cnhubert.cnhubert_base_path = cnhubert_base_path
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
ssl_model = cnhubert.get_model()
if is_half:
    bert_model = bert_model.half().to(device)
    ssl_model = ssl_model.half().to(device)
else:
    bert_model = bert_model.to(device)
    ssl_model = ssl_model.to(device)
change_gpt_sovits_weights(gpt_path = gpt_path, sovits_path = sovits_path)

resample_transform_dict={}

spec_min = -12
spec_max = 2
def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1
def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min
mel_fn=lambda x: mel_spectrogram_torch(x, **{
    "n_fft": 1024,
    "win_size": 1024,
    "hop_size": 256,
    "num_mels": 100,
    "sampling_rate": 24000,
    "fmin": 0,
    "fmax": None,
    "center": False
})


sr_model=None
def audio_sr(audio,sr):
    global sr_model
    if sr_model==None:
        from tools.audio_sr import AP_BWE
        try:
            sr_model=AP_BWE(device,DictToAttrRecursive)
        except FileNotFoundError:
            logger.info("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载")
            return audio.cpu().detach().numpy(),sr
    return sr_model(audio,sr)


        
speaker_list = {} # FIXME 封一个Speaker类

global hz # 采样率? NOTE 在哪用了？为什么这个也要全局？
hz = 50

# --------------------------------
# 接口部分
# --------------------------------
app = FastAPI()

@app.post("/set_model")
async def set_model(request: Request):
    json_post_raw = await request.json()
    return change_gpt_sovits_weights(
        gpt_path = json_post_raw.get("gpt_model_path"), 
        sovits_path = json_post_raw.get("sovits_model_path")
    )


@app.get("/set_model")
async def set_model(
        gpt_model_path: str = None,
        sovits_model_path: str = None,
):
    return change_gpt_sovits_weights(gpt_path = gpt_model_path, sovits_path = sovits_model_path)


@app.post("/control")
async def control(request: Request):
    json_post_raw = await request.json()
    return handle_control(json_post_raw.get("command"))


@app.get("/control")
async def control(command: str = None):
    return handle_control(command)


@app.post("/change_refer")
async def change_refer(request: Request):
    json_post_raw = await request.json()
    return handle_change(
        json_post_raw.get("refer_wav_path"),
        json_post_raw.get("prompt_text"),
        json_post_raw.get("prompt_language")
    )


@app.get("/change_refer")
async def change_refer(
        refer_wav_path: str = None,
        prompt_text: str = None,
        prompt_language: str = None
):
    return handle_change(refer_wav_path, prompt_text, prompt_language)


@app.post("/")
async def tts_endpoint(request: Request):
    json_post_raw = await request.json()
    return handle(
        json_post_raw.get("refer_wav_path"),
        json_post_raw.get("prompt_text"),
        json_post_raw.get("prompt_language"),
        json_post_raw.get("text"),
        json_post_raw.get("text_language"),
        json_post_raw.get("cut_punc"),
        json_post_raw.get("top_k", 15),
        json_post_raw.get("top_p", 1.0),
        json_post_raw.get("temperature", 1.0),
        json_post_raw.get("speed", 1.0),
        json_post_raw.get("inp_refs", []),
        json_post_raw.get("sample_steps", 32),
        json_post_raw.get("if_sr", False) 
    )


@app.get("/")
async def tts_endpoint(
        refer_wav_path: str = None,
        prompt_text: str = None,
        prompt_language: str = None,
        text: str = None,
        text_language: str = None,
        cut_punc: str = None,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        speed: float = 1.0,
        inp_refs: list = Query(default=[]),
        sample_steps: int = 32,
        if_sr: bool = False
):
    return handle(refer_wav_path, prompt_text, prompt_language, text, text_language, cut_punc, top_k, top_p, temperature, speed, inp_refs, sample_steps, if_sr)


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port, workers=1)
