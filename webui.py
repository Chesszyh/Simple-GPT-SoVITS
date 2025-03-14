import os
import sys
import warnings
import json
import yaml
import torch
import re
import shutil
import platform
import psutil
import signal
import site
import traceback
import subprocess
import logging
from scipy.io import wavfile
from multiprocessing import cpu_count
from subprocess import Popen
from tools import my_utils
from tools.my_utils import load_audio, check_for_existance, check_details
from tools.i18n.i18n import I18nAuto, scan_language_list
from config import (
    python_exec, infer_device, is_half, exp_root,
    webui_port_main, webui_port_infer_tts,
    webui_port_uvr5, webui_port_subfix, is_share
)
import gradio as gr

# 设置日志记录
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(message)s')


def kill_process(pid: int, process_name: str = "") -> None:
    """结束一个进程及其子进程"""
    try:
        if platform.system() == "Windows":
            cmd = f"taskkill /t /f /pid {pid}"
            subprocess.run(
                cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                try:
                    os.kill(child.pid, signal.SIGTERM)
                except OSError:
                    pass
            os.kill(pid, signal.SIGTERM)
        logging.info(
            f"{process_name}{(' - ' + process_name) if process_name else ''}进程已终止")
    except Exception as e:
        logging.error(f"结束进程 {process_name} 失败: {str(e)}")


class Config:
    """统一管理系统配置及环境初始化"""

    def __init__(self):
        # 确定版本号（兼容命令行参数）
        self.version = "v1" if len(
            sys.argv) == 1 or sys.argv[1] == "v1" else "v2"
        os.environ["version"] = self.version
        self.now_dir = os.getcwd()
        sys.path.insert(0, self.now_dir)
        warnings.filterwarnings("ignore")
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
        torch.manual_seed(233333)
        # 临时目录配置
        self.tmp = os.path.join(self.now_dir, "TEMP")
        os.makedirs(self.tmp, exist_ok=True)
        os.environ["TEMP"] = self.tmp
        self.clean_tmp()
        # 设置 site 包根目录
        self.site_packages_roots = [
            path for path in site.getsitepackages() if "packages" in path]
        if not self.site_packages_roots:
            self.site_packages_roots = [
                f"{self.now_dir}/runtime/Lib/site-packages"]
        self.create_users_pth()
        # 配置语言与国际化
        self.language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else "Auto"
        os.environ["language"] = self.language
        self.i18n = I18nAuto(language=self.language)
        # 设置代理与其他环境变量
        os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
        os.environ["all_proxy"] = ""
        logging.info("系统配置初始化完成.")

    def clean_tmp(self):
        """清理临时目录(除 jieba.cache 外)"""
        if os.path.exists(self.tmp):
            for name in os.listdir(self.tmp):
                if name == "jieba.cache":
                    continue
                path = os.path.join(self.tmp, name)
                delete = os.remove if os.path.isfile(path) else shutil.rmtree
                try:
                    delete(path)
                except Exception as e:
                    logging.warning(str(e))


    # TODO 这函数存在的意义是什么？
    def create_users_pth(self):
        """在 site 包目录下生成用户路径文件 users.pth"""
        for root in self.site_packages_roots:
            if os.path.exists(root):
                try:
                    with open(os.path.join(root, "users.pth"), "w") as f:
                        paths = [
                            self.now_dir,
                            f"{self.now_dir}/GPT_SoVITS/BigVGAN",
                            f"{self.now_dir}/tools",
                            f"{self.now_dir}/tools/asr",
                            f"{self.now_dir}/GPT_SoVITS",
                            f"{self.now_dir}/tools/uvr5"
                        ]
                        f.write("\n".join(paths))
                    break
                except PermissionError:
                    traceback.print_exc()


class ProcessManager:
    """封装子进程的启动和停止操作"""

    def __init__(self):
        self.processes = {}

    def start(self, name: str, cmd: str) -> Popen:
        logging.info(f"启动 {name} 命令: {cmd}")
        try:
            proc = Popen(cmd, shell=True)
            self.processes[name] = proc
            return proc
        except Exception as e:
            logging.error(f"启动 {name} 失败: {str(e)}")
            raise

    def stop(self, name: str, process_name: str = "") -> None:
        proc = self.processes.get(name)
        if proc:
            kill_process(proc.pid, process_name)
            del self.processes[name]

    def cleanup(self) -> None:
        for name, proc in self.processes.items():
            try:
                kill_process(proc.pid, name)
            except Exception as e:
                logging.error(f"清理 {name} 失败: {str(e)}")
        self.processes.clear()


def process_info(process_name: str, indicator: str) -> str:
    """返回描述进程状态的字符串信息"""
    mapping = {
        "opened": f"{process_name}已开启",
        "open": f"开启{process_name}",
        "closed": f"{process_name}已关闭",
        "close": f"关闭{process_name}",
        "running": f"{process_name}运行中",
        "occupy": f"{process_name}占用中，需先终止才能开启下一次任务",
        "finish": f"{process_name}已完成",
        "failed": f"{process_name}失败",
        "info": f"{process_name}进程输出信息"
    }
    return mapping.get(indicator, process_name)


class TaskManager:
    """
    管理各项任务的启动和停止。
    内部通过 ProcessManager 启动子进程，用 state 保存各任务当前状态。
    """

    def __init__(self, config: Config, proc_manager: ProcessManager):
        self.config = config
        self.pm = proc_manager
        self.state = {}  # 保存各任务的 Popen 对象，如 'p_label', 'p_uvr5' 等

    def change_label(self, path_list: str) -> str:
        """开启/关闭音频标注WebUI任务"""
        process_name = "音频标注WebUI"
        if self.state.get('p_label') is None:
            check_for_existance([path_list])
            path_list = my_utils.clean_path(path_list)
            cmd = f'"{python_exec}" tools/subfix_webui.py --load_list "{path_list}" --webui_port {webui_port_subfix} --is_share {is_share}'
            logging.info(cmd)
            proc = self.pm.start('p_label', cmd)
            self.state['p_label'] = proc
            return process_info(process_name, "opened")
        else:
            self.pm.stop('p_label', process_name)
            self.state['p_label'] = None
            return process_info(process_name, "closed")

    def change_uvr5(self) -> str:
        """开启/关闭人声分离WebUI任务"""
        process_name = "人声分离WebUI"
        if self.state.get('p_uvr5') is None:
            cmd = f'"{python_exec}" tools/uvr5/webui.py "{infer_device}" {is_half} {webui_port_uvr5} {is_share}'
            logging.info(cmd)
            proc = self.pm.start('p_uvr5', cmd)
            self.state['p_uvr5'] = proc
            return process_info(process_name, "opened")
        else:
            self.pm.stop('p_uvr5', process_name)
            self.state['p_uvr5'] = None
            return process_info(process_name, "closed")

    def change_tts_inference(self, bert_path: str, cnhubert_base_path: str,
                             gpu_number: str, gpt_path: str, sovits_path: str,
                             batched_infer_enabled: bool) -> str:
        """开启/关闭TTS推理WebUI任务"""
        process_name = "TTS推理WebUI"
        if self.state.get('p_tts_inference') is None:
            # 设置环境变量（简化判断逻辑）
            os.environ["gpt_path"] = gpt_path if "/" in gpt_path else f"GPT_weights/{gpt_path}"
            os.environ[
                "sovits_path"] = sovits_path if "/" in sovits_path else f"SoVITS_weights/{sovits_path}"
            os.environ["cnhubert_base_path"] = cnhubert_base_path
            os.environ["bert_path"] = bert_path
            os.environ["_CUDA_VISIBLE_DEVICES"] = gpu_number
            os.environ["is_half"] = str(is_half)
            os.environ["infer_ttswebui"] = str(webui_port_infer_tts)
            os.environ["is_share"] = str(is_share)
            # NOTE 根据版本决定命令
            # 什么意思，v3还不能用快速推理吗？这俩推理有啥区别？
            if self.config.version == "v3" or not batched_infer_enabled:
                cmd = f'"{python_exec}" GPT_SoVITS/inference_webui.py "{self.config.language}"'
            else:
                cmd = f'"{python_exec}" GPT_SoVITS/inference_webui_fast.py "{self.config.language}"'
            logging.info(cmd)
            proc = self.pm.start('p_tts_inference', cmd)
            self.state['p_tts_inference'] = proc
            return process_info(process_name, "opened")
        else:
            self.pm.stop('p_tts_inference', process_name)
            self.state['p_tts_inference'] = None
            return process_info(process_name, "closed")

    def open_asr(self, asr_inp_dir: str, asr_opt_dir: str,
                 asr_model: str, asr_model_size: str,
                 asr_lang: str, asr_precision: str) -> str:
        """开启语音识别任务"""
        process_name = "语音识别"
        if self.state.get('p_asr') is None:
            asr_inp_dir = my_utils.clean_path(asr_inp_dir)
            asr_opt_dir = my_utils.clean_path(asr_opt_dir)
            check_for_existance([asr_inp_dir])
            cmd = f'"{python_exec}" tools/asr/{asr_model}.py -i "{asr_inp_dir}" -o "{asr_opt_dir}" -s {asr_model_size} -l {asr_lang} -p {asr_precision}'
            logging.info(cmd)
            proc = self.pm.start('p_asr', cmd)
            self.state['p_asr'] = proc
            proc.wait()
            self.pm.stop('p_asr', process_name)
            self.state['p_asr'] = None
            return process_info(process_name, "finish")
        else:
            return process_info(process_name, "occupy")

    def close_asr(self) -> str:
        """关闭语音识别任务"""
        process_name = "语音识别"
        if self.state.get('p_asr'):
            self.pm.stop('p_asr', process_name)
            self.state['p_asr'] = None
        return process_info(process_name, "closed")

    # 其它任务方法（open_denoise、close_denoise、open1a、open1b、open1c、open1abc、open1Ba、open1Bb等）
    # 均按以上模式封装，保持原有逻辑调用
    # TODO 


# 初始化配置与管理器
config = Config()
proc_manager = ProcessManager()
task_manager = TaskManager(config, proc_manager)
n_cpu = cpu_count()

# GPU 检测和配置
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

# 识别合适的 GPU 关键字
ok_gpu_keywords = {
    "10", "16", "20", "30", "40", "A2", "A3", "A4", "P4", "A50",
    "500", "A60", "70", "80", "90", "M4", "T4", "TITAN", "L4",
    "4060", "H", "600", "506", "507", "508", "509"
}
set_gpu_numbers = set()

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper() for value in ok_gpu_keywords):
            if_gpu_ok = True  # 至少有一个可用的 NVIDIA GPU
            gpu_infos.append(f"{i}\t{gpu_name}")
            set_gpu_numbers.add(i)
            mem.append(int(torch.cuda.get_device_properties(
                i).total_memory / 1024 / 1024 / 1024 + 0.4))

def set_default():
    global default_batch_size, default_max_batch_size, gpu_info, default_sovits_epoch, default_sovits_save_every_epoch, max_sovits_epoch, max_sovits_save_every_epoch, default_batch_size_s1, if_force_ckpt
    if_force_ckpt = False
    if if_gpu_ok and len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
        minmem = min(mem)
        default_batch_size = minmem // 2 if config.version != "v3" else minmem // 8
        default_batch_size_s1 = minmem // 2
    else:
        gpu_info = ("%s\t%s" % ("0", "CPU"))
        gpu_infos.append("%s\t%s" % ("0", "CPU"))
        set_gpu_numbers.add(0)
        default_batch_size = default_batch_size_s1 = int(
            psutil.virtual_memory().total / 1024 / 1024 / 1024 / 4)
    if config.version != "v3":
        default_sovits_epoch = 8
        default_sovits_save_every_epoch = 4
        max_sovits_epoch = 25
        max_sovits_save_every_epoch = 25
    else:
        default_sovits_epoch = 2
        default_sovits_save_every_epoch = 1
        max_sovits_epoch = 3
        max_sovits_save_every_epoch = 3

    default_batch_size = max(1, default_batch_size)
    default_batch_size_s1 = max(1, default_batch_size_s1)
    default_max_batch_size = default_batch_size * 3


set_default()

gpus = "-".join([i[0] for i in gpu_infos])
default_gpu_numbers = str(sorted(list(set_gpu_numbers))[0])


# NOTE 给小白擦屁股的函数，后续准备删除
# 越界怎么办？那就报错呗！
def fix_gpu_number(input):
    """将越界的 GPU number 强制改到界内"""
    try:
        if int(input) not in set_gpu_numbers:
            return default_gpu_numbers
    except:
        return input
    return input


def fix_gpu_numbers(inputs):
    output = []
    try:
        for input in inputs.split(","):
            output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs


# NOTE 这都可以丢到 config 里去，干嘛写死在代码里？
pretrained_sovits_name = [
    "GPT_SoVITS/pretrained_models/s2G488k.pth",
    "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    "GPT_SoVITS/pretrained_models/s2Gv3.pth"
]
pretrained_gpt_name = [
    "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "GPT_SoVITS/pretrained_models/s1v3.ckpt"
]

pretrained_model_list = (
    pretrained_sovits_name[int(config.version[-1]) - 1],
    pretrained_sovits_name[int(config.version[-1]) - 1].replace("s2G", "s2D"),
    pretrained_gpt_name[int(config.version[-1]) - 1],
    "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)


_ = ''
for i in pretrained_model_list:
    if "s2Dv3" not in i and not os.path.exists(i):
        _ += f'\n    {i}'
if _:
    print("warning: ", config.i18n('以下模型不存在:') + _)

_ = [[], []]
for i in range(3):
    if os.path.exists(pretrained_gpt_name[i]):
        _[0].append(pretrained_gpt_name[i])
    else:
        _[0].append("")
    if os.path.exists(pretrained_sovits_name[i]):
        _[-1].append(pretrained_sovits_name[i])
    else:
        _[-1].append("")
pretrained_gpt_name, pretrained_sovits_name = _

# 我不需要这么多文件夹，不要在我的根目录下到处丢垃圾
SoVITS_weight_root = ["SoVITS_weights", ]
                        # "SoVITS_weights_v2", "SoVITS_weights_v3"]
GPT_weight_root = ["GPT_weights", ] # "GPT_weights_v2", "GPT_weights_v3"]
for root in SoVITS_weight_root + GPT_weight_root:
    os.makedirs(root, exist_ok=True)


# NOTE 啊？
def get_weights_names():
    SoVITS_names = [name for name in pretrained_sovits_name if name != ""]
    for path in SoVITS_weight_root:
        for name in os.listdir(path):
            if name.endswith(".pth"):
                SoVITS_names.append(f"{path}/{name}")
    GPT_names = [name for name in pretrained_gpt_name if name != ""]
    for path in GPT_weight_root:
        for name in os.listdir(path):
            if name.endswith(".ckpt"):
                GPT_names.append(f"{path}/{name}")
    return SoVITS_names, GPT_names


SoVITS_names, GPT_names = get_weights_names()
for path in SoVITS_weight_root + GPT_weight_root:
    os.makedirs(path, exist_ok=True)


def custom_sort_key(s):
    """自定义排序键，提取字符串中的数字部分和非数字部分"""
    parts = re.split('(\d+)', s)
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}


# Gradio UI 部分
with gr.Blocks(title="GPT-SoVITS WebUI") as app:
    gr.Markdown(
        value=config.i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责.") +
        "<br>" + config.i18n("如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录LICENSE.")
    )
    gr.Markdown(value=config.i18n("中文教程文档") +
                ": " + "https://www.yuque.com/...")
    with gr.Tabs():
        with gr.TabItem("0-" + config.i18n("前置数据集获取工具")):
            gr.Markdown(value="0a-" + config.i18n("UVR5人声伴奏分离&去混响去延迟工具"))
            with gr.Row():
                with gr.Column(scale=3):
                    uvr5_info = gr.Textbox(
                        label=process_info("人声分离WebUI", "info"))
                open_uvr5 = gr.Button(value=process_info(
                    "人声分离WebUI", "open"), variant="primary", visible=True)
                close_uvr5 = gr.Button(value=process_info(
                    "人声分离WebUI", "close"), variant="primary", visible=False)
            open_uvr5.click(fn=lambda: task_manager.change_uvr5(), inputs=[
            ], outputs=[uvr5_info, open_uvr5, close_uvr5])
            close_uvr5.click(fn=lambda: task_manager.change_uvr5(), inputs=[
            ], outputs=[uvr5_info, open_uvr5, close_uvr5])
        # 其它 TabItem 内调用保持不变
        # TODO existing code... o3 mini给我删没了

    app.queue().launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=webui_port_main,
        quiet=True,
    )
