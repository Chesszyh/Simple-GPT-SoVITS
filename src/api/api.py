import argparse
import os
import re
import sys
import signal
import logging
import subprocess
from time import time as ttime
from io import BytesIO
from typing import Any, Dict, Tuple, List, Optional

import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ...existing import code...
from text.LangSegmenter import LangSegmenter
from feature_extractor import cnhubert
from module.models import SynthesizerTrn, SynthesizerTrnV3
from peft import LoraConfig, PeftModel, get_peft_model
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch, mel_spectrogram_torch
from tools.my_utils import load_audio
import config as global_config

# ------------------------------
# 配置与初始化相关函数
# ------------------------------


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数并返回参数对象。"""
    parser = argparse.ArgumentParser(description="GPT-SoVITS api")
    parser.add_argument("-s", "--sovits_path", type=str,
                        default=global_config.Config().sovits_path, help="SoVITS模型路径")
    parser.add_argument("-g", "--gpt_path", type=str,
                        default=global_config.Config().gpt_path, help="GPT模型路径")
    parser.add_argument("-dr", "--default_refer_path",
                        type=str, default="", help="默认参考音频路径")
    parser.add_argument("-dt", "--default_refer_text",
                        type=str, default="", help="默认参考音频文本")
    parser.add_argument("-dl", "--default_refer_language",
                        type=str, default="", help="默认参考音频语种")
    parser.add_argument("-d", "--device", type=str,
                        default=global_config.Config().infer_device, help="cuda / cpu")
    parser.add_argument("-a", "--bind_addr", type=str,
                        default="0.0.0.0", help="绑定地址")
    parser.add_argument("-p", "--port", type=int,
                        default=global_config.Config().api_port, help="端口号")
    parser.add_argument("-fp", "--full_precision",
                        action="store_true", default=False, help="使用全精度")
    parser.add_argument("-hp", "--half_precision",
                        action="store_true", default=False, help="使用半精度")
    parser.add_argument("-sm", "--stream_mode", type=str,
                        default="close", help="流式返回模式, close / normal / keepalive")
    parser.add_argument("-mt", "--media_type", type=str,
                        default="wav", help="音频编码格式, wav / ogg / aac")
    parser.add_argument("-st", "--sub_type", type=str,
                        default="int16", help="音频数据类型, int16 / int32")
    parser.add_argument("-cp", "--cut_punc", type=str,
                        default="", help="文本切分符号设定")
    parser.add_argument("-hb", "--hubert_path", type=str,
                        default=global_config.Config().cnhubert_path, help="覆盖config.cnhubert_path")
    parser.add_argument("-b", "--bert_path", type=str,
                        default=global_config.Config().bert_path, help="覆盖config.bert_path")
    return parser.parse_args()


def init_logger() -> logging.Logger:
    """初始化日志系统并返回logger对象"""
    logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
    return logging.getLogger('uvicorn')


class DefaultRefer:
    def __init__(self, path: str, text: str, language: str) -> None:
        self.path: str = path
        self.text: str = text
        self.language: str = language

    def is_ready(self) -> bool:
        return all((self.path, self.text, self.language))


def init_models(args: argparse.Namespace, logger: logging.Logger, device: str, is_half: bool) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    初始化bert、hubert、ssl以及tokenizer等模型，返回各模型对象。
    注意：保持原有加载逻辑和设备设置不变。
    """
    cnhubert.cnhubert_base_path = args.hubert_path
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(args.bert_path)
    ssl_model = cnhubert.get_model()
    if is_half:
        bert_model = bert_model.half().to(device)
        ssl_model = ssl_model.half().to(device)
    else:
        bert_model = bert_model.to(device)
        ssl_model = ssl_model.to(device)
    return tokenizer, bert_model, ssl_model, None, None, None  # 后续保持原有接口调用


def init_app() -> Tuple[FastAPI, Dict[str, Any]]:
    """
    整体初始化，包括参数解析、日志、模型和必要全局变量封装到 context 字典中。
    """
    args = parse_arguments()
    logger = init_logger()
    config_obj = global_config.Config()
    # 判断精度设置，并设置 is_half
    is_half = config_obj.is_half
    if args.full_precision:
        is_half = False
    if args.half_precision:
        is_half = True
    # 流式和编码等模式设置
    stream_mode = "normal" if args.stream_mode.lower() in [
        "normal", "n"] else "close"
    media_type = args.media_type.lower() if args.media_type.lower(
    ) in ["aac", "ogg"] else ("wav" if stream_mode == "close" else "ogg")
    is_int32 = True if args.sub_type.lower() == 'int32' else False

    # 初始化默认参考
    default_refer = DefaultRefer(
        args.default_refer_path, args.default_refer_text, args.default_refer_language)
    if not default_refer.is_ready():
        logger.info("未指定默认参考音频")
    else:
        logger.info(f"默认参考音频路径: {default_refer.path}")

    # 模型路径检查及fallback
    if args.sovits_path == "":
        args.sovits_path = config_obj.pretrained_sovits_path
        logger.warning(f"未指定SoVITS模型路径, fallback后当前值: {args.sovits_path}")
    if args.gpt_path == "":
        args.gpt_path = config_obj.pretrained_gpt_path
        logger.warning(f"未指定GPT模型路径, fallback后当前值: {args.gpt_path}")

    # 设置全局运行环境变量
    device = args.device

    tokenizer, bert_model, ssl_model, _, _, _ = init_models(
        args, logger, device, is_half)

    # 封装上下文，后续各函数可传递
    context = {
        "args": args,
        "logger": logger,
        "device": device,
        "is_half": is_half,
        "default_refer": default_refer,
        "stream_mode": stream_mode,
        "media_type": media_type,
        "is_int32": is_int32,
        "tokenizer": tokenizer,
        "bert_model": bert_model,
        "ssl_model": ssl_model,
        "speaker_list": {},   # 保存 speaker 对象
        "dict_language": {     # 语言映射表，保持原有逻辑
            "中文": "all_zh",
            "粤语": "all_yue",
            "英文": "en",
            "日文": "all_ja",
            "韩文": "all_ko",
            "中英混合": "zh",
            "粤英混合": "yue",
            "日英混合": "ja",
            "韩英混合": "ko",
            "多语种混合": "auto",
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
        },
    }
    return FastAPI(), context

# ------------------------------
# 业务逻辑函数（保留原有功能，增加类型提示和错误处理）
# ------------------------------


def change_gpt_sovits_weights(gpt_path: str, sovits_path: str, context: Dict[str, Any]) -> JSONResponse:
    """
    加载GPT与SoVITS模型权重，更新context['speaker_list']中"default"的speaker对象。
    """
    try:
        # ...调用原有的 get_gpt_weights 与 get_sovits_weights...
        gpt = get_gpt_weights(gpt_path)
        sovits = get_sovits_weights(sovits_path)
    except Exception as e:
        return JSONResponse({"code": 400, "message": str(e)}, status_code=400)
    context["speaker_list"]["default"] = Speaker(
        name="default", gpt=gpt, sovits=sovits)
    return JSONResponse({"code": 0, "message": "Success"}, status_code=200)


def handle_control(command: str) -> Any:
    """处理控制命令，用于重启或退出程序。"""
    if command == "restart":
        os.execl(sys.executable, sys.executable, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
    return JSONResponse({"code": 0, "message": f"Command {command} executed"}, status_code=200)


def handle_change(path: Optional[str], text: Optional[str], language: Optional[str], context: Dict[str, Any]) -> JSONResponse:
    """更新默认参考音频的配置信息。"""
    default_refer: DefaultRefer = context["default_refer"]
    if not all([path, text, language]):
        return JSONResponse({"code": 400, "message": '缺少参数: "path", "text", "language"'}, status_code=400)
    default_refer.path = path
    default_refer.text = text
    default_refer.language = language
    context["logger"].info(f"当前默认参考音频路径: {default_refer.path}")
    context["logger"].info(f"当前默认参考音频文本: {default_refer.text}")
    context["logger"].info(f"当前默认参考音频语种: {default_refer.language}")
    context["logger"].info(f"is_ready: {default_refer.is_ready()}")
    return JSONResponse({"code": 0, "message": "Success"}, status_code=200)


def handle(*args, **kwargs) -> StreamingResponse:
    # ...existing code中的 handle 函数保持原逻辑...
    return StreamingResponse(get_tts_wav(*args, **kwargs), media_type="audio/" + kwargs.get("media_type", "wav"))

# 其它业务逻辑函数如 get_gpt_weights、get_sovits_weights、get_bert_feature 等均保持不变
# ...existing code...


# ------------------------------
# FastAPI 路由绑定
# ------------------------------
app, context = init_app()


@app.post("/set_model")
async def set_model_post(request: Request) -> Any:
    json_post = await request.json()
    return change_gpt_sovits_weights(
        gpt_path=json_post.get("gpt_model_path"),
        sovits_path=json_post.get("sovits_model_path"),
        context=context
    )


@app.get("/set_model")
async def set_model_get(gpt_model_path: str = None, sovits_model_path: str = None) -> Any:
    return change_gpt_sovits_weights(gpt_model_path, sovits_model_path, context)


@app.post("/control")
async def control_post(request: Request) -> Any:
    json_post = await request.json()
    return handle_control(json_post.get("command"))


@app.get("/control")
async def control_get(command: str = None) -> Any:
    return handle_control(command)


@app.post("/change_refer")
async def change_refer_post(request: Request) -> Any:
    json_post = await request.json()
    return handle_change(
        json_post.get("refer_wav_path"),
        json_post.get("prompt_text"),
        json_post.get("prompt_language"),
        context
    )


@app.get("/change_refer")
async def change_refer_get(refer_wav_path: str = None, prompt_text: str = None, prompt_language: str = None) -> Any:
    return handle_change(refer_wav_path, prompt_text, prompt_language, context)


@app.post("/")
async def tts_endpoint_post(request: Request) -> Any:
    json_post = await request.json()
    return handle(
        json_post.get("refer_wav_path"),
        json_post.get("prompt_text"),
        json_post.get("prompt_language"),
        json_post.get("text"),
        json_post.get("text_language"),
        json_post.get("cut_punc"),
        json_post.get("top_k", 15),
        json_post.get("top_p", 1.0),
        json_post.get("temperature", 1.0),
        json_post.get("speed", 1.0),
        json_post.get("inp_refs", []),
        json_post.get("sample_steps", 32),
        json_post.get("if_sr", False),
        media_type=context["media_type"],
    )


@app.get("/")
async def tts_endpoint_get(
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
    inp_refs: List[str] = Query(default=[]),
    sample_steps: int = 32,
    if_sr: bool = False
) -> Any:
    return handle(
        refer_wav_path,
        prompt_text,
        prompt_language,
        text,
        text_language,
        cut_punc,
        top_k,
        top_p,
        temperature,
        speed,
        inp_refs,
        sample_steps,
        if_sr,
        media_type=context["media_type"],
    )

# ------------------------------
# 主入口
# ------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host=context["args"].bind_addr,
                port=context["args"].port, workers=1)
