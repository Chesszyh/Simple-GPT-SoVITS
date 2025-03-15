import os
import sys
import traceback
import signal
import argparse
import subprocess
import wave
import logging
from typing import Dict, List, Optional, Generator, Tuple, Any, Union, Literal
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field
import uvicorn

# 初始化项目路径
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))

# 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GPT-SoVITS-API")

try:
    from tools.i18n.i18n import I18nAuto
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
except ImportError as e:
    logger.error(f"导入组件失败: {e}")
    logger.error("请确保已安装所有依赖并且目录结构正确")
    sys.exit(1)

# 初始化国际化及参数
i18n = I18nAuto()
cut_method_names = get_cut_method_names()


class TTSRequestModel(BaseModel):
    """TTS请求参数模型"""
    text: str = Field(..., description="要合成的文本")
    text_lang: str = Field(..., description="文本语言")
    ref_audio_path: str = Field(..., description="参考音频路径")
    aux_ref_audio_paths: Optional[List[str]] = Field(None, description="多说话人音色融合的辅助参考音频路径")
    prompt_text: str = Field("", description="参考音频的提示文本")
    prompt_lang: str = Field(..., description="参考音频的提示文本语言")
    top_k: int = Field(5, description="top k采样")
    top_p: float = Field(1, description="top p采样")
    temperature: float = Field(1, description="采样温度")
    text_split_method: str = Field("cut5", description="文本分割方法")
    batch_size: int = Field(1, description="推理批处理大小")
    batch_threshold: float = Field(0.75, description="批处理分割阈值")
    split_bucket: bool = Field(True, description="是否将批次分割成多个桶")
    speed_factor: float = Field(1.0, description="控制生成音频的速度")
    fragment_interval: float = Field(0.3, description="控制音频片段的间隔")
    seed: int = Field(-1, description="随机种子")
    media_type: str = Field("wav", description="输出音频格式")
    streaming_mode: bool = Field(False, description="是否返回流式响应")
    parallel_infer: bool = Field(True, description="是否使用并行推理")
    repetition_penalty: float = Field(1.35, description="T2S模型的重复惩罚")


class AudioProcessor:
    """音频处理类，处理各种音频格式的转换"""
    
    @staticmethod
    def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
        """将音频数据打包为OGG格式
        
        Args:
            io_buffer: 输出缓冲区
            data: 音频数据
            rate: 采样率
            
        Returns:
            打包后的OGG格式音频
        """
        with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
            audio_file.write(data)
        return io_buffer

    @staticmethod
    def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
        """将音频数据打包为RAW格式
        
        Args:
            io_buffer: 输出缓冲区
            data: 音频数据
            rate: 采样率
            
        Returns:
            打包后的RAW格式音频
        """
        io_buffer.write(data.tobytes())
        return io_buffer

    @staticmethod
    def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
        """将音频数据打包为WAV格式
        
        Args:
            io_buffer: 输出缓冲区
            data: 音频数据
            rate: 采样率
            
        Returns:
            打包后的WAV格式音频
        """
        io_buffer = BytesIO()
        sf.write(io_buffer, data, rate, format='wav')
        return io_buffer

    @staticmethod
    def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
        """将音频数据打包为AAC格式
        
        Args:
            io_buffer: 输出缓冲区
            data: 音频数据
            rate: 采样率
            
        Returns:
            打包后的AAC格式音频
        """
        try:
            process = subprocess.Popen([
                'ffmpeg',
                '-f', 's16le',
                '-ar', str(rate),
                '-ac', '1',
                '-i', 'pipe:0',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-vn',
                '-f', 'adts',
                'pipe:1'
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            out, err = process.communicate(input=data.tobytes())
            if process.returncode != 0:
                logger.warning(f"AAC转换警告: {err.decode('utf-8', errors='ignore')}")
            io_buffer.write(out)
            return io_buffer
        except Exception as e:
            logger.error(f"AAC转换失败: {e}")
            # 如果AAC转换失败，回退到WAV格式
            return AudioProcessor.pack_wav(io_buffer, data, rate)
    
    @staticmethod
    def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str) -> BytesIO:
        """根据指定的媒体类型打包音频数据
        
        Args:
            io_buffer: 输出缓冲区
            data: 音频数据
            rate: 采样率
            media_type: 媒体类型 ('ogg', 'aac', 'wav', 'raw')
            
        Returns:
            打包后的音频
        """
        if media_type == "ogg":
            io_buffer = AudioProcessor.pack_ogg(io_buffer, data, rate)
        elif media_type == "aac":
            io_buffer = AudioProcessor.pack_aac(io_buffer, data, rate)
        elif media_type == "wav":
            io_buffer = AudioProcessor.pack_wav(io_buffer, data, rate)
        else:  # raw
            io_buffer = AudioProcessor.pack_raw(io_buffer, data, rate)
        io_buffer.seek(0)
        return io_buffer

    @staticmethod
    def wave_header_chunk(frame_input: bytes = b"", channels: int = 1, 
                         sample_width: int = 2, sample_rate: int = 32000) -> bytes:
        """创建WAV文件头
        
        Args:
            frame_input: 音频帧数据
            channels: 通道数
            sample_width: 采样位宽
            sample_rate: 采样率
            
        Returns:
            WAV文件头数据
        """
        wav_buf = BytesIO()
        with wave.open(wav_buf, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(frame_input)
        
        wav_buf.seek(0)
        return wav_buf.read()


class GPTSoVITSAPI:
    """GPT-SoVITS API服务类"""
    
    def __init__(self, config_path: str):
        """初始化API服务
        
        Args:
            config_path: TTS配置文件路径
        """
        self.config_path = config_path
        self.tts_config = TTS_Config(config_path)
        self.tts_pipeline = TTS(self.tts_config)
        self.app = FastAPI()
        self.audio_processor = AudioProcessor()
        self._setup_routes()
        logger.info(f"GPT-SoVITS API 初始化完成，配置: {self.tts_config}")
    
    def _setup_routes(self) -> None:
        """设置API路由"""
        # TTS路由
        self.app.get("/tts")(self.tts_get_endpoint)
        self.app.post("/tts")(self.tts_post_endpoint)
        
        # 控制路由
        self.app.get("/control")(self.control_endpoint)
        
        # 模型切换路由
        self.app.get("/set_gpt_weights")(self.set_gpt_weights_endpoint)
        self.app.get("/set_sovits_weights")(self.set_sovits_weights_endpoint)
        self.app.get("/set_refer_audio")(self.set_refer_audio_endpoint)
    
    def _check_tts_params(self, req: Dict[str, Any]) -> Optional[JSONResponse]:
        """检查TTS请求参数
        
        Args:
            req: 请求参数字典
            
        Returns:
            如果参数有错误，返回错误响应；否则返回None
        """
        text = req.get("text", "")
        text_lang = req.get("text_lang", "")
        ref_audio_path = req.get("ref_audio_path", "")
        streaming_mode = req.get("streaming_mode", False)
        media_type = req.get("media_type", "wav")
        prompt_lang = req.get("prompt_lang", "")
        text_split_method = req.get("text_split_method", "cut5")

        # 必需参数检查
        if not ref_audio_path:
            return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
        if not text:
            return JSONResponse(status_code=400, content={"message": "text is required"})
        
        # 语言支持检查
        if not text_lang:
            return JSONResponse(status_code=400, content={"message": "text_lang is required"})
        elif text_lang.lower() not in self.tts_config.languages:
            return JSONResponse(status_code=400, content={
                "message": f"text_lang: {text_lang} is not supported in version {self.tts_config.version}"
            })
            
        if not prompt_lang:
            return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
        elif prompt_lang.lower() not in self.tts_config.languages:
            return JSONResponse(status_code=400, content={
                "message": f"prompt_lang: {prompt_lang} is not supported in version {self.tts_config.version}"
            })
        
        # 媒体类型检查
        if media_type not in ["wav", "raw", "ogg", "aac"]:
            return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
        elif media_type == "ogg" and not streaming_mode:
            return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})
        
        # 分割方法检查
        if text_split_method not in cut_method_names:
            return JSONResponse(status_code=400, content={
                "message": f"text_split_method:{text_split_method} is not supported"
            })

        return None
    
    async def tts_handle(self, req: Dict[str, Any]) -> Union[Response, JSONResponse, StreamingResponse]:
        """处理TTS请求
        
        Args:
            req: 请求参数字典
            
        Returns:
            音频响应或错误信息
        """
        streaming_mode = req.get("streaming_mode", False)
        return_fragment = req.get("return_fragment", False)
        media_type = req.get("media_type", "wav")

        # 参数检查
        check_res = self._check_tts_params(req)
        if check_res is not None:
            return check_res

        if streaming_mode or return_fragment:
            req["return_fragment"] = True
        
        try:
            tts_generator = self.tts_pipeline.run(req)
            
            if streaming_mode:
                # 构建流式响应
                def streaming_generator(tts_generator: Generator, media_type: str):
                    if media_type == "wav":
                        yield self.audio_processor.wave_header_chunk()
                        media_type = "raw"
                    for sr, chunk in tts_generator:
                        yield self.audio_processor.pack_audio(BytesIO(), chunk, sr, media_type).getvalue()
                
                return StreamingResponse(
                    streaming_generator(tts_generator, media_type),
                    media_type=f"audio/{media_type}"
                )
            else:
                # 一次性响应
                sr, audio_data = next(tts_generator)
                audio_data = self.audio_processor.pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
                return Response(audio_data, media_type=f"audio/{media_type}")
        except Exception as e:
            logger.error(f"TTS处理失败: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=400, 
                content={"message": "TTS处理失败", "exception": str(e)}
            )
    
    def handle_control(self, command: str) -> None:
        """处理控制命令
        
        Args:
            command: 控制命令 (restart, exit)
        """
        logger.info(f"接收到控制命令: {command}")
        if command == "restart":
            logger.info("重启服务...")
            os.execl(sys.executable, sys.executable, *sys.argv)
        elif command == "exit":
            logger.info("退出服务...")
            os.kill(os.getpid(), signal.SIGTERM)
            sys.exit(0)
    
    async def tts_get_endpoint(self,
                        text: str = None,
                        text_lang: str = None,
                        ref_audio_path: str = None,
                        aux_ref_audio_paths: List[str] = None,
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
                        repetition_penalty: float = 1.35
                        ) -> Union[Response, JSONResponse, StreamingResponse]:
        """GET方式的TTS端点
        
        Returns:
            音频响应或错误信息
        """
        req = {
            "text": text,
            "text_lang": text_lang.lower() if text_lang else None,
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": aux_ref_audio_paths,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang.lower() if prompt_lang else None,
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
            "repetition_penalty": float(repetition_penalty)
        }
        return await self.tts_handle(req)
    
    async def tts_post_endpoint(self, request: TTSRequestModel) -> Union[Response, JSONResponse, StreamingResponse]:
        """POST方式的TTS端点
        
        Args:
            request: TTS请求模型
            
        Returns:
            音频响应或错误信息
        """
        req = request.dict()
        return await self.tts_handle(req)
    
    async def control_endpoint(self, command: str = None) -> JSONResponse:
        """控制端点
        
        Args:
            command: 控制命令
            
        Returns:
            成功/失败响应
        """
        if not command:
            return JSONResponse(status_code=400, content={"message": "command is required"})
        try:
            self.handle_control(command)
            return JSONResponse(status_code=200, content={"message": "success"})
        except Exception as e:
            logger.error(f"执行控制命令失败: {e}")
            return JSONResponse(status_code=500, content={"message": f"执行控制命令失败: {str(e)}"})
    
    async def set_refer_audio_endpoint(self, refer_audio_path: str = None) -> JSONResponse:
        """设置参考音频端点
        
        Args:
            refer_audio_path: 参考音频路径
            
        Returns:
            成功/失败响应
        """
        try:
            self.tts_pipeline.set_ref_audio(refer_audio_path)
            return JSONResponse(status_code=200, content={"message": "success"})
        except Exception as e:
            logger.error(f"设置参考音频失败: {e}")
            return JSONResponse(
                status_code=400, 
                content={"message": "设置参考音频失败", "exception": str(e)}
            )
    
    async def set_gpt_weights_endpoint(self, weights_path: str = None) -> JSONResponse:
        """设置GPT权重端点
        
        Args:
            weights_path: 模型权重路径
            
        Returns:
            成功/失败响应
        """
        try:
            if not weights_path:
                return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
            
            self.tts_pipeline.init_t2s_weights(weights_path)
            logger.info(f"GPT模型切换成功: {weights_path}")
            return JSONResponse(status_code=200, content={"message": "success"})
        except Exception as e:
            logger.error(f"切换GPT权重失败: {e}")
            return JSONResponse(
                status_code=400, 
                content={"message": "切换GPT权重失败", "exception": str(e)}
            )
    
    async def set_sovits_weights_endpoint(self, weights_path: str = None) -> JSONResponse:
        """设置SoVITS权重端点
        
        Args:
            weights_path: 模型权重路径
            
        Returns:
            成功/失败响应
        """
        try:
            if not weights_path:
                return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
            
            self.tts_pipeline.init_vits_weights(weights_path)
            logger.info(f"SoVITS模型切换成功: {weights_path}")
            return JSONResponse(status_code=200, content={"message": "success"})
        except Exception as e:
            logger.error(f"切换SoVITS权重失败: {e}")
            return JSONResponse(
                status_code=400, 
                content={"message": "切换SoVITS权重失败", "exception": str(e)}
            )
    
    def run(self, host: str, port: int) -> None:
        """运行API服务
        
        Args:
            host: 监听地址
            port: 监听端口
        """
        try:
            logger.info(f"启动API服务: {host}:{port}")
            if host == 'None':
                host = None  # 使用None表示监听所有地址（包括IPv6）
            uvicorn.run(app=self.app, host=host, port=port, workers=1)
        except Exception as e:
            logger.error(f"API服务运行失败: {e}", exc_info=True)
            os.kill(os.getpid(), signal.SIGTERM)
            sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GPT-SoVITS API")
    parser.add_argument("-c", "--tts_config", type=str, 
                        default="GPT_SoVITS/configs/tts_infer.yaml", 
                        help="tts_infer路径")
    parser.add_argument("-a", "--bind_addr", type=str, 
                        default="127.0.0.1", 
                        help="监听地址，默认: 127.0.0.1")
    parser.add_argument("-p", "--port", type=int, 
                        default="9880", 
                        help="监听端口，默认: 9880")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        config_path = args.tts_config
        host = args.bind_addr
        port = args.port
        
        if not config_path or config_path == "":
            config_path = "GPT_SoVITS/configs/tts_infer.yaml"
        
        api = GPTSoVITSAPI(config_path)
        api.run(host, port)
    except Exception as e:
        logger.error(f"程序运行失败: {e}", exc_info=True)
        traceback.print_exc()
        sys.exit(1)
