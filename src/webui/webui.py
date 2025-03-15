import re
from config import python_exec, infer_device, is_half, exp_root, webui_port_main, webui_port_infer_tts, webui_port_uvr5, webui_port_subfix, is_share
import torch
import yaml
import json
from tools.asr.config import asr_dict
import gradio as gr
from multiprocessing import cpu_count
from tools.my_utils import load_audio, check_for_existance, check_details
from scipy.io import wavfile
from tools.i18n.i18n import I18nAuto, scan_language_list
from subprocess import Popen
import subprocess
import pdb
import shutil
from tools import my_utils
import traceback
import site
import signal
import psutil
import platform
import warnings
import os
import sys
if len(sys.argv) == 1:
    sys.argv.append('v2')
version = "v1"if sys.argv[1] == "v1" else "v2"
os.environ["version"] = version
now_dir = os.getcwd()
sys.path.insert(0, now_dir)
warnings.filterwarnings("ignore")
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
torch.manual_seed(233333)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
if (os.path.exists(tmp)):
    for name in os.listdir(tmp):
        if (name == "jieba.cache"):
            continue
        path = "%s/%s" % (tmp, name)
        delete = os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass
site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if (site_packages_roots == []):
    site_packages_roots = ["%s/runtime/Lib/site-packages" % now_dir]
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/users.pth" % (site_packages_root), "w") as f:
                f.write(
                    # "%s\n%s/runtime\n%s/tools\n%s/tools/asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    "%s\n%s/GPT_SoVITS/BigVGAN\n%s/tools\n%s/tools/asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    % (now_dir, now_dir, now_dir, now_dir, now_dir, now_dir)
                )
            break
        except PermissionError as e:
            traceback.print_exc()
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else "Auto"
os.environ["language"] = language
i18n = I18nAuto(language=language)
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # 当遇到mps不支持的步骤时使用cpu
try:
    import gradio.analytics as analytics
    analytics.version_check = lambda: None
except:
    ...
n_cpu = cpu_count()

ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

# 判断是否有能用来训练和加速推理的N卡
ok_gpu_keywords = {"10", "16", "20", "30", "40", "A2", "A3", "A4", "P4", "A50", "500", "A60",
                   "70", "80", "90", "M4", "T4", "TITAN", "L4", "4060", "H", "600", "506", "507", "508", "509"}
set_gpu_numbers = set()
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper()for value in ok_gpu_keywords):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            set_gpu_numbers.add(i)
            mem.append(int(torch.cuda.get_device_properties(
                i).total_memory / 1024 / 1024 / 1024 + 0.4))


def set_default():
    global default_batch_size, default_max_batch_size, gpu_info, default_sovits_epoch, default_sovits_save_every_epoch, max_sovits_epoch, max_sovits_save_every_epoch, default_batch_size_s1, if_force_ckpt
    if_force_ckpt = False
    if if_gpu_ok and len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
        minmem = min(mem)
        default_batch_size = minmem // 2 if version != "v3"else minmem//8
        default_batch_size_s1 = minmem // 2
    else:
        gpu_info = ("%s\t%s" % ("0", "CPU"))
        gpu_infos.append("%s\t%s" % ("0", "CPU"))
        set_gpu_numbers.add(0)
        default_batch_size = default_batch_size_s1 = int(
            psutil.virtual_memory().total / 1024 / 1024 / 1024 / 4)
    if version != "v3":
        default_sovits_epoch = 8
        default_sovits_save_every_epoch = 4
        max_sovits_epoch = 25  # 40
        max_sovits_save_every_epoch = 25  # 10
    else:
        default_sovits_epoch = 2
        default_sovits_save_every_epoch = 1
        max_sovits_epoch = 3  # 40
        max_sovits_save_every_epoch = 3  # 10

    default_batch_size = max(1, default_batch_size)
    default_batch_size_s1 = max(1, default_batch_size_s1)
    default_max_batch_size = default_batch_size * 3


set_default()

gpus = "-".join([i[0] for i in gpu_infos])
default_gpu_numbers = str(sorted(list(set_gpu_numbers))[0])


def fix_gpu_number(input):  # 将越界的number强制改到界内
    try:
        if (int(input)not in set_gpu_numbers):
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


pretrained_sovits_name = ["GPT_SoVITS/pretrained_models/s2G488k.pth",
                          "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth", "GPT_SoVITS/pretrained_models/s2Gv3.pth"]
pretrained_gpt_name = ["GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
                       "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", "GPT_SoVITS/pretrained_models/s1v3.ckpt"]

pretrained_model_list = (pretrained_sovits_name[int(version[-1])-1], pretrained_sovits_name[int(version[-1])-1].replace("s2G", "s2D"), pretrained_gpt_name[int(
    version[-1])-1], "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large", "GPT_SoVITS/pretrained_models/chinese-hubert-base")

_ = ''
for i in pretrained_model_list:
    if "s2Dv3" not in i and os.path.exists(i) == False:
        _ += f'\n    {i}'
if _:
    print("warning: ", i18n('以下模型不存在:') + _)

_ = [[], []]
for i in range(3):
    if os.path.exists(pretrained_gpt_name[i]):
        _[0].append(pretrained_gpt_name[i])
    else:
        _[0].append("")  # 没有下pretrained模型的，说不定他们是想自己从零训底模呢
    if os.path.exists(pretrained_sovits_name[i]):
        _[-1].append(pretrained_sovits_name[i])
    else:
        _[-1].append("")
pretrained_gpt_name, pretrained_sovits_name = _

SoVITS_weight_root = ["SoVITS_weights",
                      "SoVITS_weights_v2", "SoVITS_weights_v3"]
GPT_weight_root = ["GPT_weights", "GPT_weights_v2", "GPT_weights_v3"]
for root in SoVITS_weight_root+GPT_weight_root:
    os.makedirs(root, exist_ok=True)


def get_weights_names():
    SoVITS_names = [name for name in pretrained_sovits_name if name != ""]
    for path in SoVITS_weight_root:
        for name in os.listdir(path):
            if name.endswith(".pth"):
                SoVITS_names.append("%s/%s" % (path, name))
    GPT_names = [name for name in pretrained_gpt_name if name != ""]
    for path in GPT_weight_root:
        for name in os.listdir(path):
            if name.endswith(".ckpt"):
                GPT_names.append("%s/%s" % (path, name))
    return SoVITS_names, GPT_names


SoVITS_names, GPT_names = get_weights_names()
for path in SoVITS_weight_root+GPT_weight_root:
    os.makedirs(path, exist_ok=True)


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}


p_label = None
p_uvr5 = None
p_asr = None
p_denoise = None
p_tts_inference = None


def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass


system = platform.system()


def kill_process(pid, process_name=""):
    if (system == "Windows"):
        cmd = "taskkill /t /f /pid %s" % pid
        # os.system(cmd)
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    else:
        kill_proc_tree(pid)
    print(process_name + i18n("进程已终止"))


def process_info(process_name="", indicator=""):
    if indicator == "opened":
        return process_name + i18n("已开启")
    elif indicator == "open":
        return i18n("开启") + process_name
    elif indicator == "closed":
        return process_name + i18n("已关闭")
    elif indicator == "close":
        return i18n("关闭") + process_name
    elif indicator == "running":
        return process_name + i18n("运行中")
    elif indicator == "occupy":
        return process_name + i18n("占用中") + "," + i18n("需先终止才能开启下一次任务")
    elif indicator == "finish":
        return process_name + i18n("已完成")
    elif indicator == "failed":
        return process_name + i18n("失败")
    elif indicator == "info":
        return process_name + i18n("进程输出信息")
    else:
        return process_name


process_name_subfix = i18n("音频标注WebUI")


def change_label(path_list):
    global p_label
    if p_label is None:
        check_for_existance([path_list])
        path_list = my_utils.clean_path(path_list)
        cmd = '"%s" tools/subfix_webui.py --load_list "%s" --webui_port %s --is_share %s' % (
            python_exec, path_list, webui_port_subfix, is_share)
        print(process_info(process_name_subfix, "opened"))
        print(cmd)
        p_label = Popen(cmd, shell=True)
    else:
        kill_process(p_label.pid, process_name_subfix)
        p_label = None
        print(process_info(process_name_subfix, "closed"))


process_name_uvr5 = i18n("人声分离WebUI")


def change_uvr5():
    global p_uvr5
    if p_uvr5 is None:
        cmd = '"%s" tools/uvr5/webui.py "%s" %s %s %s' % (
            python_exec, infer_device, is_half, webui_port_uvr5, is_share)
        print(process_info(process_name_uvr5, "opened"))
        print(cmd)
        p_uvr5 = Popen(cmd, shell=True)
    else:
        kill_process(p_uvr5.pid, process_name_uvr5)
        p_uvr5 = None
        print(process_info(process_name_uvr5, "closed"))


process_name_tts = i18n("TTS推理WebUI")


def change_tts_inference(bert_path, cnhubert_base_path, gpu_number, gpt_path, sovits_path, batched_infer_enabled):
    global p_tts_inference
    if batched_infer_enabled:
        cmd = '"%s" GPT_SoVITS/inference_webui_fast.py "%s"' % (
            python_exec, language)
    else:
        cmd = '"%s" GPT_SoVITS/inference_webui.py "%s"' % (
            python_exec, language)
    # v3暂不支持加速推理
    if version == "v3":
        cmd = '"%s" GPT_SoVITS/inference_webui.py "%s"' % (
            python_exec, language)
    if p_tts_inference is None:
        os.environ["gpt_path"] = gpt_path if "/" in gpt_path else "%s/%s" % (
            GPT_weight_root, gpt_path)
        os.environ["sovits_path"] = sovits_path if "/" in sovits_path else "%s/%s" % (
            SoVITS_weight_root, sovits_path)
        os.environ["cnhubert_base_path"] = cnhubert_base_path
        os.environ["bert_path"] = bert_path
        os.environ["_CUDA_VISIBLE_DEVICES"] = fix_gpu_number(gpu_number)
        os.environ["is_half"] = str(is_half)
        os.environ["infer_ttswebui"] = str(webui_port_infer_tts)
        os.environ["is_share"] = str(is_share)
        print(process_info(process_name_tts, "opened"))
        print(cmd)
        p_tts_inference = Popen(cmd, shell=True)
    else:
        kill_process(p_tts_inference.pid, process_name_tts)
        p_tts_inference = None
        print(process_info(process_name_tts, "closed"))


process_name_asr = i18n("语音识别")


def open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang, asr_precision):
    global p_asr
    if p_asr is None:
        asr_inp_dir = my_utils.clean_path(asr_inp_dir)
        asr_opt_dir = my_utils.clean_path(asr_opt_dir)
        check_for_existance([asr_inp_dir])
        cmd = f'"{python_exec}" tools/asr/{asr_dict[asr_model]["path"]}'
        cmd += f' -i "{asr_inp_dir}"'
        cmd += f' -o "{asr_opt_dir}"'
        cmd += f' -s {asr_model_size}'
        cmd += f' -l {asr_lang}'
        cmd += f" -p {asr_precision}"
        output_file_name = os.path.basename(asr_inp_dir)
        output_folder = asr_opt_dir or "output/asr_opt"
        output_file_path = os.path.abspath(
            f'{output_folder}/{output_file_name}.list')
        print(process_info(process_name_asr, "opened"))
        print(cmd)
        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr = None
        print(process_info(process_name_asr, "finish"))
    else:
        print(process_info(process_name_asr, "occupy"))


def close_asr():
    global p_asr
    if p_asr is not None:
        kill_process(p_asr.pid, process_name_asr)
        p_asr = None
    print(process_info(process_name_asr, "closed"))


process_name_denoise = i18n("语音降噪")


def open_denoise(denoise_inp_dir, denoise_opt_dir):
    global p_denoise
    if (p_denoise == None):
        denoise_inp_dir = my_utils.clean_path(denoise_inp_dir)
        denoise_opt_dir = my_utils.clean_path(denoise_opt_dir)
        check_for_existance([denoise_inp_dir])
        cmd = '"%s" tools/cmd-denoise.py -i "%s" -o "%s" -p %s' % (
            python_exec, denoise_inp_dir, denoise_opt_dir, "float16"if is_half == True else "float32")

        print(process_info(process_name_denoise, "opened"))
        print(cmd)
        p_denoise = Popen(cmd, shell=True)
        p_denoise.wait()
        p_denoise = None
        print(process_info(process_name_denoise, "finish"))
    else:
        print(process_info(process_name_denoise, "occupy"))


def close_denoise():
    global p_denoise
    if p_denoise is not None:
        kill_process(p_denoise.pid, process_name_denoise)
        p_denoise = None
    print(process_info(process_name_denoise, "closed"))


p_train_SoVITS = None
process_name_sovits = i18n("SoVITS训练")


def open1Ba(batch_size, total_epoch, exp_name, text_low_lr_rate, if_save_latest, if_save_every_weights, save_every_epoch, gpu_numbers1Ba, pretrained_s2G, pretrained_s2D, if_grad_ckpt, lora_rank):
    global p_train_SoVITS
    if (p_train_SoVITS == None):
        with open("GPT_SoVITS/configs/s2.json")as f:
            data = f.read()
            data = json.loads(data)
        s2_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s2_%s" % (s2_dir, version), exist_ok=True)
        if check_for_existance([s2_dir], is_train=True):
            check_details([s2_dir], is_train=True)
        if (is_half == False):
            data["train"]["fp16_run"] = False
            batch_size = max(1, batch_size//2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["train"]["text_low_lr_rate"] = text_low_lr_rate
        data["train"]["pretrained_s2G"] = pretrained_s2G
        data["train"]["pretrained_s2D"] = pretrained_s2D
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["save_every_epoch"] = save_every_epoch
        data["train"]["gpu_numbers"] = gpu_numbers1Ba
        data["train"]["grad_ckpt"] = if_grad_ckpt
        data["train"]["lora_rank"] = lora_rank
        data["model"]["version"] = version
        data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
        data["save_weight_dir"] = SoVITS_weight_root[int(version[-1])-1]
        data["name"] = exp_name
        data["version"] = version
        tmp_config_path = "%s/tmp_s2.json" % tmp
        with open(tmp_config_path, "w")as f:
            f.write(json.dumps(data))
        if version in ["v1", "v2"]:
            cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"' % (
                python_exec, tmp_config_path)
        else:
            cmd = '"%s" GPT_SoVITS/s2_train_v3_lora.py --config "%s"' % (
                python_exec, tmp_config_path)
        print(process_info(process_name_sovits, "opened"))
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS = None
        print(process_info(process_name_sovits, "finish"))
    else:
        print(process_info(process_name_sovits, "occupy"))


def close1Ba():
    global p_train_SoVITS
    if p_train_SoVITS is not None:
        kill_process(p_train_SoVITS.pid, process_name_sovits)
        p_train_SoVITS = None
    print(process_info(process_name_sovits, "closed"))


p_train_GPT = None
process_name_gpt = i18n("GPT训练")


def open1Bb(batch_size, total_epoch, exp_name, if_dpo, if_save_latest, if_save_every_weights, save_every_epoch, gpu_numbers, pretrained_s1):
    global p_train_GPT
    if (p_train_GPT == None):
        with open("GPT_SoVITS/configs/s1longer.yaml"if version == "v1"else "GPT_SoVITS/configs/s1longer-v2.yaml")as f:
            data = f.read()
            data = yaml.load(data, Loader=yaml.FullLoader)
        s1_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
        if check_for_existance([s1_dir], is_train=True):
            check_details([s1_dir], is_train=True)
        if (is_half == False):
            data["train"]["precision"] = "32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["pretrained_s1"] = pretrained_s1
        data["train"]["save_every_n_epoch"] = save_every_epoch
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_dpo"] = if_dpo
        data["train"]["half_weights_save_dir"] = GPT_weight_root[int(
            version[-1])-1]
        data["train"]["exp_name"] = exp_name
        data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
        data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
        data["output_dir"] = "%s/logs_s1_%s" % (s1_dir, version)
        # data["version"]=version

        os.environ["_CUDA_VISIBLE_DEVICES"] = fix_gpu_numbers(
            gpu_numbers.replace("-", ","))
        os.environ["hz"] = "25hz"
        tmp_config_path = "%s/tmp_s1.yaml" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" ' % (
            python_exec, tmp_config_path)
        print(process_info(process_name_gpt, "opened"))
        print(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT = None
        print(process_info(process_name_gpt, "finish"))
    else:
        print(process_info(process_name_gpt, "occupy"))


def close1Bb():
    global p_train_GPT
    if p_train_GPT is not None:
        kill_process(p_train_GPT.pid, process_name_gpt)
        p_train_GPT = None
    print(process_info(process_name_gpt, "closed"))


ps_slice = []
process_name_slice = i18n("语音切分")


def open_slice(inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, n_parts):
    global ps_slice
    inp = my_utils.clean_path(inp)
    opt_root = my_utils.clean_path(opt_root)
    check_for_existance([inp])
    if (os.path.exists(inp) == False):
        print(i18n("输入路径不存在"))
        return
    if os.path.isfile(inp):
        n_parts = 1
    elif os.path.isdir(inp):
        pass
    else:
        print(i18n("输入路径存在但不可用"))
        return
    if (ps_slice == []):
        for i_part in range(n_parts):
            cmd = '"%s" tools/slice_audio.py "%s" "%s" %s %s %s %s %s %s %s %s %s''' % (
                python_exec, inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, i_part, n_parts)
            print(cmd)
            p = Popen(cmd, shell=True)
            ps_slice.append(p)
        print(process_info(process_name_slice, "opened"))
        for p in ps_slice:
            p.wait()
        ps_slice = []
        print(process_info(process_name_slice, "finish"))
    else:
        print(process_info(process_name_slice, "occupy"))


def close_slice():
    global ps_slice
    if (ps_slice != []):
        for p_slice in ps_slice:
            try:
                kill_process(p_slice.pid, process_name_slice)
            except:
                traceback.print_exc()
        ps_slice = []
    print(process_info(process_name_slice, "closed"))


ps1a = []
process_name_1a = i18n("文本分词与特征提取")


def open1a(inp_text, inp_wav_dir, exp_name, gpu_numbers, bert_pretrained_dir):
    global ps1a
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    if (ps1a == []):
        opt_dir = "%s/%s" % (exp_root, exp_name)
        config = {
            "inp_text": inp_text,
            "inp_wav_dir": inp_wav_dir,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "bert_pretrained_dir": bert_pretrained_dir,
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                    "is_half": str(is_half)
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py' % python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1a.append(p)
        print(process_info(process_name_1a, "running"))
        for p in ps1a:
            p.wait()
        opt = []
        for i_part in range(all_parts):
            txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
            with open(txt_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(txt_path)
        path_text = "%s/2-name2text.txt" % opt_dir
        with open(path_text, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1a = []
        if len("".join(opt)) > 0:
            print(process_info(process_name_1a, "finish"))
        else:
            print(process_info(process_name_1a, "failed"))
    else:
        print(process_info(process_name_1a, "occupy"))


def close1a():
    global ps1a
    if ps1a != []:
        for p1a in ps1a:
            try:
                kill_process(p1a.pid, process_name_1a)
            except:
                traceback.print_exc()
        ps1a = []
    print(process_info(process_name_1a, "closed"))


ps1b = []
process_name_1b = i18n("语音自监督特征提取")


def open1b(inp_text, inp_wav_dir, exp_name, gpu_numbers, ssl_pretrained_dir):
    global ps1b
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    if (ps1b == []):
        config = {
            "inp_text": inp_text,
            "inp_wav_dir": inp_wav_dir,
            "exp_name": exp_name,
            "opt_dir": "%s/%s" % (exp_root, exp_name),
            "cnhubert_base_dir": ssl_pretrained_dir,
            "is_half": str(is_half)
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py' % python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1b.append(p)
        print(process_info(process_name_1b, "running"))
        for p in ps1b:
            p.wait()
        ps1b = []
        print(process_info(process_name_1b, "finish"))
    else:
        print(process_info(process_name_1b, "occupy"))


def close1b():
    global ps1b
    if (ps1b != []):
        for p1b in ps1b:
            try:
                kill_process(p1b.pid, process_name_1b)
            except:
                traceback.print_exc()
        ps1b = []
    print(process_info(process_name_1b, "closed"))


ps1c = []
process_name_1c = i18n("语义Token提取")


def open1c(inp_text, exp_name, gpu_numbers, pretrained_s2G_path):
    global ps1c
    inp_text = my_utils.clean_path(inp_text)
    if check_for_existance([inp_text, ''], is_dataset_processing=True):
        check_details([inp_text, ''], is_dataset_processing=True)
    if (ps1c == []):
        opt_dir = "%s/%s" % (exp_root, exp_name)
        config = {
            "inp_text": inp_text,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "pretrained_s2G": pretrained_s2G_path,
            "s2config_path": "GPT_SoVITS/configs/s2.json",
            "is_half": str(is_half)
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py' % python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1c.append(p)
        print(process_info(process_name_1c, "running"))
        for p in ps1c:
            p.wait()
        opt = ["item_name\tsemantic_audio"]
        path_semantic = "%s/6-name2semantic.tsv" % opt_dir
        for i_part in range(all_parts):
            semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
            with open(semantic_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(semantic_path)
        with open(path_semantic, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1c = []
        print(process_info(process_name_1c, "finish"))
    else:
        print(process_info(process_name_1c, "occupy"))


def close1c():
    global ps1c
    if (ps1c != []):
        for p1c in ps1c:
            try:
                kill_process(p1c.pid, process_name_1c)
            except:
                traceback.print_exc()
        ps1c = []
    print(process_info(process_name_1c, "closed"))


ps1abc = []
process_name_1abc = i18n("训练集格式化一键三连")


def open1abc(inp_text, inp_wav_dir, exp_name, gpu_numbers1a, gpu_numbers1Ba, gpu_numbers1c, bert_pretrained_dir, ssl_pretrained_dir, pretrained_s2G_path):
    global ps1abc
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    if (ps1abc == []):
        opt_dir = "%s/%s" % (exp_root, exp_name)
        try:
            # 1a
            path_text = "%s/2-name2text.txt" % opt_dir
            if (os.path.exists(path_text) == False or (os.path.exists(path_text) == True and len(open(path_text, "r", encoding="utf8").read().strip("\n").split("\n")) < 2)):
                config = {
                    "inp_text": inp_text,
                    "inp_wav_dir": inp_wav_dir,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "bert_pretrained_dir": bert_pretrained_dir,
                    "is_half": str(is_half)
                }
                gpu_names = gpu_numbers1a.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                print(i18n("进度") + ": 1A-Doing")
                for p in ps1abc:
                    p.wait()

                opt = []
                # txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                for i_part in range(all_parts):
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                assert len("".join(opt)) > 0, process_info(
                    process_name_1a, "failed")
            print(i18n("进度") + ": 1A-Done")
            ps1abc = []
            # 1b
            config = {
                "inp_text": inp_text,
                "inp_wav_dir": inp_wav_dir,
                "exp_name": exp_name,
                "opt_dir": opt_dir,
                "cnhubert_base_dir": ssl_pretrained_dir,
            }
            gpu_names = gpu_numbers1Ba.split("-")
            all_parts = len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                    }
                )
                os.environ.update(config)
                cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py' % python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            print(i18n("进度") + ": 1A-Done, 1B-Doing")
            for p in ps1abc:
                p.wait()
            print(i18n("进度") + ": 1A-Done, 1B-Done")
            ps1abc = []
            # 1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if (os.path.exists(path_semantic) == False or (os.path.exists(path_semantic) == True and os.path.getsize(path_semantic) < 31)):
                config = {
                    "inp_text": inp_text,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "pretrained_s2G": pretrained_s2G_path,
                    "s2config_path": "GPT_SoVITS/configs/s2.json",
                }
                gpu_names = gpu_numbers1c.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                print(i18n("进度") + ": 1A-Done, 1B-Done, 1C-Doing")
                for p in ps1abc:
                    p.wait()

                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (
                        opt_dir, i_part)
                    with open(semantic_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                print(i18n("进度") + ": 1A-Done, 1B-Done, 1C-Done")
            ps1abc = []
            print(process_info(process_name_1abc, "finish"))
        except:
            traceback.print_exc()
            close1abc()
            print(process_info(process_name_1abc, "failed"))
    else:
        print(process_info(process_name_1abc, "occupy"))


def close1abc():
    global ps1abc
    if (ps1abc != []):
        for p1abc in ps1abc:
            try:
                kill_process(p1abc.pid, process_name_1abc)
            except:
                traceback.print_exc()
        ps1abc = []
    print(process_info(process_name_1abc, "closed"))


def switch_version(version_):
    os.environ["version"] = version_
    global version
    version = version_
    if pretrained_sovits_name[int(version[-1])-1] != '' and pretrained_gpt_name[int(version[-1])-1] != '':
        ...
    else:
        print(i18n('未下载模型') + ": " + version.upper())
    set_default()
    return {'__type__': 'update', 'value': pretrained_sovits_name[int(version[-1])-1]}, \
        {'__type__': 'update', 'value': pretrained_sovits_name[int(version[-1])-1].replace("s2G", "s2D")}, \
        {'__type__': 'update', 'value': pretrained_gpt_name[int(version[-1])-1]}, \
        {'__type__': 'update', 'value': pretrained_gpt_name[int(version[-1])-1]}, \
        {'__type__': 'update', 'value': pretrained_sovits_name[int(version[-1])-1]}, \
        {'__type__': 'update', "value": default_batch_size, "maximum": default_max_batch_size}, \
        {'__type__': 'update', "value": default_sovits_epoch, "maximum": max_sovits_epoch}, \
        {'__type__': 'update', "value": default_sovits_save_every_epoch, "maximum": max_sovits_save_every_epoch}, \
        {'__type__': 'update', "visible": True if version != "v3"else False}, \
        {'__type__': 'update', "value": False if not if_force_ckpt else True, "interactive": True if not if_force_ckpt else False}, \
        {'__type__': 'update', "interactive": False if version == "v3" else True, "value": False}, \
        {'__type__': 'update', "visible": True if version == "v3" else False}


if os.path.exists('GPT_SoVITS/text/G2PWModel'):
    ...
else:
    cmd = '"%s" GPT_SoVITS/download.py' % python_exec
    p = Popen(cmd, shell=True)
    p.wait()


def sync(text):
    return {'__type__': 'update', 'value': text}

# 添加 CLI 封装接口


def run_inference(args):
    """
    示例：通过命令行调用 TTS 推理功能。
    必须提供 bert_path, cnhubert_base_path, gpu_number, gpt_path, sovits_path, batched_infer_enabled 参数
    """
    # 根据命令行传入参数设置环境变量
    os.environ["is_half"] = str(is_half)
    os.environ["_CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # 调用核心推理函数（注意：原函数为 generator，这里直接执行第一步操作）
    print("启动推理...")
    gen = change_tts_inference(
        args.bert, args.cnhubert, args.gpu, args.gpt, args.sovits, args.batched)
    # 直接打印所有过程信息
    for msg in gen:
        print("状态:", msg)
    print("推理进程已启动或终止。")


def run_asr(args):
    print("启动 ASR 任务...")
    # 直接调用 open_asr，参数由命令行传入
    proc_gen = open_asr(args.input_dir, args.output_dir,
                        args.model, args.size, args.lang, args.precision)
    for msg in proc_gen:
        print("状态:", msg)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Headless GPT-SoVITS 服务器接口")
    subparsers = parser.add_subparsers(
        dest="mode", required=True, help="请选择运行模式")

    # 推理模式：tts_inference
    parser_infer = subparsers.add_parser("tts_inference", help="进行 TTS 推理")
    parser_infer.add_argument(
        "--bert", type=str, required=True, help="预训练 BERT 模型路径")
    parser_infer.add_argument("--cnhubert", type=str,
                              required=True, help="预训练 CN-Hubert 模型路径")
    parser_infer.add_argument("--gpu", type=str, required=True, help="GPU 号")
    parser_infer.add_argument(
        "--gpt", type=str, required=True, help="GPT 模型路径")
    parser_infer.add_argument("--sovits", type=str,
                              required=True, help="SoVITS 模型路径")
    parser_infer.add_argument("--batched", action="store_true", help="启用并行推理")

    # ASR 模式
    parser_asr = subparsers.add_parser("asr", help="进行 ASR 推理")
    parser_asr.add_argument("--input_dir", type=str,
                            required=True, help="ASR 输入目录")
    parser_asr.add_argument("--output_dir", type=str,
                            required=True, help="ASR 输出目录")
    parser_asr.add_argument("--model", type=str,
                            required=True, help="ASR 模型名称")
    parser_asr.add_argument("--size", type=str, required=True, help="ASR 模型尺寸")
    parser_asr.add_argument("--lang", type=str, required=True, help="ASR 语言")
    parser_asr.add_argument("--precision", type=str,
                            required=True, help="ASR 精度")

    # 可按需添加其他模式（如 denoise, slice, train 等）

    args = parser.parse_args()
    if args.mode == "tts_inference":
        run_inference(args)
    elif args.mode == "asr":
        run_asr(args)
    else:
        print("未知模式:", args.mode)


if __name__ == "__main__":
    main()
