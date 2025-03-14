"""
GPT-SoVITS 配置管理模块

此模块提供项目中所有配置参数的集中管理，包括:
- 模型路径
- 设备设置
- 网络服务端口配置
- 推理参数
"""
import os
import sys
from typing import Optional, Literal, Union, Dict, Any
import torch


class Config:
    """配置类，提供对项目配置的集中访问"""
    
    def __init__(self):
        """初始化配置项，读取环境变量和检测系统状态"""
        # 模型路径配置
        self.sovits_path: str = "neuro-sama/AI-Neuro-TTS_e8_s888.pth"
        self.gpt_path: str = "neuro-sama/AI-Neuro-TTS-e15.ckpt"
        self.cnhubert_path: str = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        self.bert_path: str = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        self.pretrained_sovits_path: str = "GPT_SoVITS/pretrained_models/s2G488k.pth"
        self.pretrained_gpt_path: str = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
        
        # 环境变量读取
        is_half_str = os.environ.get("is_half", "True")
        self.is_half: bool = is_half_str.lower() == 'true'
        
        is_share_str = os.environ.get("is_share", "False")
        self.is_share: bool = is_share_str.lower() == 'true'
        
        # 运行环境配置
        self.exp_root: str = "logs"
        self.python_exec: str = sys.executable or "python"
        
        # 设备配置
        self.infer_device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 根据GPU自动调整half精度
        if self.infer_device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            low_vram_gpus = [
                "16" in gpu_name and "V100" not in gpu_name.upper(),
                "P40" in gpu_name.upper(),
                "P10" in gpu_name.upper(),
                "1060" in gpu_name,
                "1070" in gpu_name,
                "1080" in gpu_name
            ]
            if any(low_vram_gpus):
                self.is_half = False
        
        # CPU不支持half精度
        if self.infer_device == "cpu":
            self.is_half = False
        
        # Web服务端口配置
        self.webui_port_main: int = 9874
        self.webui_port_uvr5: int = 9873
        self.webui_port_infer_tts: int = 9872
        self.webui_port_subfix: int = 9871
        self.api_port: int = 9880
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置导出为字典格式
        
        Returns:
            Dict[str, Any]: 包含所有配置项的字典
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __str__(self) -> str:
        """配置的字符串表示
        
        Returns:
            str: 配置内容的字符串表示
        """
        config_items = [f"{k}={v}" for k, v in self.to_dict().items()]
        return f"Config({', '.join(config_items)})"


# 创建默认配置实例供全局使用
config = Config()
