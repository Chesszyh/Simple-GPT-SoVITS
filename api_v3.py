import os
import sys
import traceback
from typing import Generator

import torch

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import Response
from fastapi.responses import JSONResponse
from fastapi import FastAPI
import uvicorn
from io import BytesIO
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from functools import lru_cache

cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 0.0.0.0")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
port = args.port
host = args.bind_addr
argv = sys.argv

APP = FastAPI()


class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    tts_infer_yaml_path: str = None
    """推理时需要加载的声音模型的yaml配置文件路径，如：GPT_SoVITS/configs/tts_infer.yaml"""


@lru_cache(maxsize=10)
def get_tts_instance(tts_config: TTS_Config) -> TTS:
    print(f"load tts config from {tts_config.configs_path}")
    return TTS(tts_config)


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    """modify from https://github.com/RVC-Boss/GPT-SoVITS/pull/894/files"""
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def check_params(req: dict, tts_config: TTS_Config):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut5")

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if (text_lang in [None, ""]):
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang.lower() not in tts_config.languages:
        return JSONResponse(status_code=400, content={"message": "text_lang is not supported"})
    if (prompt_lang in [None, ""]):
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is not supported"})
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": "media_type is not supported"})
    elif media_type == "ogg" and not streaming_mode:
        return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})

    if text_split_method not in cut_method_names:
        return JSONResponse(status_code=400,
                            content={"message": f"text_split_method:{text_split_method} is not supported"})

    return None


async def tts_handle(req: dict):
    """
    Text to speech handler.
    
    Args:
        req (dict): 
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 5,                   # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "media_type": "wav",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
                "streaming_mode": False,      # bool. whether to return a streaming response.
                "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.          
            }
    returns:
        StreamingResponse: audio stream response.
    """

    streaming_mode = req.get("streaming_mode", False)
    media_type = req.get("media_type", "wav")
    tts_infer_yaml_path = req.get("tts_infer_yaml_path", "GPT_SoVITS/configs/tts_infer.yaml")

    tts_config = TTS_Config(tts_infer_yaml_path)
    check_res = check_params(req, tts_config)
    if check_res is not None:
        return check_res

    if streaming_mode:
        req["return_fragment"] = True

    try:
        tts_instance = get_tts_instance(tts_config)

        move_to_original(tts_instance, tts_config)

        tts_generator = tts_instance.run(req)

        if streaming_mode:
            def streaming_generator(tts_generator: Generator, media_type: str):
                if media_type == "wav":
                    yield wave_header_chunk()
                    media_type = "raw"
                for sr, chunk in tts_generator:
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()
                move_to_cpu(tts_instance)

            # _media_type = f"audio/{media_type}" if not (streaming_mode and media_type in ["wav", "raw"]) else f"audio/x-{media_type}"
            return StreamingResponse(streaming_generator(tts_generator, media_type, ), media_type=f"audio/{media_type}")

        else:
            sr, audio_data = next(tts_generator)
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            move_to_cpu(tts_instance)
            return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"tts failed", "Exception": str(e)})


def move_to_cpu(tts):
    cpu_device = torch.device('cpu')
    tts.set_device(cpu_device, False)
    tts.enable_half_precision(False, False)
    print("Moved TTS models to CPU to save GPU memory.")


def move_to_original(tts: TTS, tts_config: TTS_Config):
    tts.set_device(tts_config.device, False)
    tts.enable_half_precision(tts_config.is_half, False)
    print("Moved TTS models back to original device for performance.")


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)


@APP.get("/tts")
async def tts_get_endpoint(
        text: str = None,
        text_lang: str = None,
        ref_audio_path: str = None,
        prompt_lang: str = None,
        prompt_text: str = "",
        top_k: int = 5,
        top_p: float = 1,
        temperature: float = 1,
        text_split_method: str = "cut0",
        batch_size: int = 1,
        batch_threshold: float = 0.75,
        split_bucket: bool = True,
        speed_factor: float = 1.0,
        fragment_interval: float = 0.3,
        seed: int = -1,
        media_type: str = "wav",
        streaming_mode: bool = False,
        parallel_infer: bool = True,
        repetition_penalty: float = 1.35,
        tts_infer_yaml_path: str = "GPT_SoVITS/configs/tts_infer.yaml"
):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty),
        "tts_infer_yaml_path": tts_infer_yaml_path
    }

    return await tts_handle(req)


@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)


@APP.get("/set_refer_audio")
async def set_refer_audio(refer_audio_path: str = None, tts_infer_yaml_path: str = "GPT_SoVITS/configs/tts_infer.yaml"):
    try:
        tts_config = TTS_Config(tts_infer_yaml_path)
        tts_instance = get_tts_instance(tts_config)
        tts_instance.set_ref_audio(refer_audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None, tts_infer_yaml_path: str = "GPT_SoVITS/configs/tts_infer.yaml"):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})

        tts_config = TTS_Config(tts_infer_yaml_path)
        tts_instance = get_tts_instance(tts_config)
        tts_instance.init_t2s_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"change gpt weight failed", "Exception": str(e)})

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None, tts_infer_yaml_path: str = "GPT_SoVITS/configs/tts_infer.yaml"):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})

        tts_config = TTS_Config(tts_infer_yaml_path)
        tts_instance = get_tts_instance(tts_config)
        tts_instance.init_vits_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


if __name__ == "__main__":
    try:
        uvicorn.run(APP, host=host, port=port, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
