# GPT-SoVITS API调用手册

## 1. 概述

本手册详细介绍了GPT-SoVITS系统提供的API接口，帮助开发者编写程序使AI agent调用这些API进行语音处理、训练和合成。GPT-SoVITS是一个强大的文本到语音系统，允许通过API进行自动化操作。

## 2. 环境设置

### 安装客户端

```python
pip install gradio_client
```

注意，目前需要使用`gpt-sovits`环境的gradio和gradio-client，`manus`没有gradio，并且gradio_clent版本要高很多。后续考虑合并manus和gpt-sovits环境。

NOTE sovits需要3.9，manus需要3.12，conda环境合并可能有问题，后续可考虑：1. 使用Docker 2. 换其他包管理器。<https://chatgpt.com/share/67d694de-4f94-800c-8f28-7873b0174832>

### 连接服务器

```python
from gradio_client import Client
client = Client("http://localhost:9874/")
```

## 3. API功能分类

### 3.1 音频预处理

| API | 描述 |
|-----|------|
| `/change_uvr5`, `/change_uvr5_1` | 打开人声分离WebUI |
| `/open_slice`, `/close_slice` | 音频切片处理 |
| `/open_denoise`, `/close_denoise` | 音频降噪处理 |

### 3.2 语音识别与标注

| API | 描述 |
|-----|------|
| `/change_lang_choices` | 更改ASR语言选项 |
| `/change_size_choices` | 更改ASR模型大小选项 |
| `/change_precision_choices` | 更改计算精度选项 |
| `/open_asr`, `/close_asr` | 执行语音识别 |
| `/change_label`, `/change_label_1` | 变更标签文件 |

### 3.3 特征提取

| API | 描述 |
|-----|------|
| `/open1a`, `/close1a` | 文本标记与BERT特征提取 |
| `/open1b`, `/close1b` | 语音SSL特征提取 |
| `/open1c`, `/close1c` | 语义token提取 |
| `/open1abc`, `/close1abc` | 一键格式化训练集 |

### 3.4 模型训练

| API | 描述 |
|-----|------|
| `/open1Ba`, `/close1Ba` | SoVITS训练 |
| `/open1Bb`, `/close1Bb` | GPT训练 |

### 3.5 TTS推理

| API | 描述 |
|-----|------|
| `/change_choices` | 更改模型权重选项 |
| `/change_tts_inference`, `/change_tts_inference_1` | 配置TTS推理参数 |
| `/switch_version` | 切换模型版本 |

## 4. 详细API文档

### 4.1 音频预处理

#### 4.1.1 人声分离

```python
# 打开人声分离WebUI
result = client.predict(api_name="/change_uvr5")
```

#### 4.1.2 音频切片

```python
# 执行音频切片
result = client.predict(
    inp="输入文件或目录路径",                # 音频切片输入
    opt_root="output/slicer_opt",          # 输出目录
    threshold="-34",                       # 噪声门限阈值
    min_length="4000",                     # 最小片段长度
    min_interval="300",                    # 音频切割最小间隔
    hop_size="10",                         # FO跳跃大小
    max_sil_kept="500",                    # 保留的最大静音长度
    _max=0.9,                              # 归一化后的响度倍增器
    alpha=0.25,                            # 归一化音频混合到数据集的比例
    n_parts=4,                             # 音频切片使用的CPU线程数
    api_name="/open_slice"
)

# 关闭音频切片进程
result = client.predict(api_name="/close_slice")
```

#### 4.1.3 音频降噪

```python
# 执行音频降噪
result = client.predict(
    denoise_inp_dir="输入目录路径",          # 输入目录
    denoise_opt_dir="output/denoise_opt",   # 输出目录
    api_name="/open_denoise"
)

# 关闭音频降噪进程
result = client.predict(api_name="/close_denoise")
```

### 4.2 语音识别与标注

#### 4.2.1 ASR配置选项

```python
# 更改ASR语言选项
result = client.predict(
    key="达摩 ASR (中文)",  # 可选值: '达摩 ASR (中文)', 'Faster Whisper (多语种)'
    api_name="/change_lang_choices"
)

# 更改ASR模型大小选项
result = client.predict(
    key="达摩 ASR (中文)",  # 可选值: '达摩 ASR (中文)', 'Faster Whisper (多语种)'
    api_name="/change_size_choices"
)

# 更改计算精度选项
result = client.predict(
    key="达摩 ASR (中文)",  # 可选值: '达摩 ASR (中文)', 'Faster Whisper (多语种)'
    api_name="/change_precision_choices"
)
```

#### 4.2.2 执行语音识别

```python
# 执行语音识别
result = client.predict(
    asr_inp_dir="D:\\GPT-SoVITS\\raw\\xxx",  # 输入目录
    asr_opt_dir="output/asr_opt",            # 输出目录
    asr_model="达摩 ASR (中文)",               # ASR模型
    asr_model_size="large",                  # 模型大小
    asr_lang="zh",                           # ASR语言
    asr_precision="float32",                 # 计算精度
    api_name="/open_asr"
)

# 关闭语音识别进程
result = client.predict(api_name="/close_asr")
```

#### 4.2.3 标签处理

```python
# 更改标签文件
result = client.predict(
    path_list="D:\\RVC1006\\GPT-SoVITS\\raw\\xxx.list",  # 标签文件路径
    api_name="/change_label"
)
```

### 4.3 特征提取

#### 4.3.1 文本特征提取

```python
# 文本标记与BERT特征提取
result = client.predict(
    inp_text="D:\\RVC1006\\GPT-SoVITS\\raw\\xxx.list",  # 文本标记文件
    inp_wav_dir="音频数据集目录",                        # 音频数据集目录
    exp_name="xxx",                                    # 实验/模型名称
    gpu_numbers="0-0",                                # GPU编号，用-分隔
    bert_pretrained_dir="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large", # 预训练中文BERT模型路径
    api_name="/open1a"
)

# 关闭特征提取进程
result = client.predict(api_name="/close1a")
```

#### 4.3.2 语音SSL特征提取

```python
# 语音SSL特征提取
result = client.predict(
    inp_text="D:\\RVC1006\\GPT-SoVITS\\raw\\xxx.list",  # 文本标记文件
    inp_wav_dir="音频数据集目录",                        # 音频数据集目录
    exp_name="xxx",                                    # 实验/模型名称
    gpu_numbers="0-0",                                # GPU编号，用-分隔
    ssl_pretrained_dir="GPT_SoVITS/pretrained_models/chinese-hubert-base", # 预训练SSL模型路径
    api_name="/open1b"
)

# 关闭SSL特征提取进程
result = client.predict(api_name="/close1b")
```

#### 4.3.3 语义token提取

```python
# 语义token提取
result = client.predict(
    inp_text="D:\\RVC1006\\GPT-SoVITS\\raw\\xxx.list",  # 文本标记文件
    exp_name="xxx",                                    # 实验/模型名称
    gpu_numbers="0-0",                                # GPU编号，用-分隔
    pretrained_s2G_path="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth", # 预训练SoVITS-G模型路径
    api_name="/open1c"
)

# 关闭语义token提取进程
result = client.predict(api_name="/close1c")
```

#### 4.3.4 一键格式化训练集

```python
# 一键格式化训练集
result = client.predict(
    inp_text="D:\\RVC1006\\GPT-SoVITS\\raw\\xxx.list",  # 文本标记文件
    inp_wav_dir="音频数据集目录",                        # 音频数据集目录
    exp_name="xxx",                                    # 实验/模型名称
    gpu_numbers1a="0-0",                              # 1a步骤的GPU编号
    gpu_numbers1Ba="0-0",                             # 1b步骤的GPU编号
    gpu_numbers1c="0-0",                              # 1c步骤的GPU编号
    bert_pretrained_dir="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large", # 预训练中文BERT模型路径
    ssl_pretrained_dir="GPT_SoVITS/pretrained_models/chinese-hubert-base", # 预训练SSL模型路径
    pretrained_s2G_path="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth", # 预训练SoVITS-G模型路径
    api_name="/open1abc"
)

# 关闭一键格式化训练集进程
result = client.predict(api_name="/close1abc")
```

### 4.4 模型训练

#### 4.4.1 SoVITS训练

```python
# 执行SoVITS训练
result = client.predict(
    batch_size=4,                        # 每个GPU的批量大小
    total_epoch=8,                       # 总训练轮数
    exp_name="xxx",                      # 实验/模型名称
    text_low_lr_rate=0.4,                # 文本模型学习率权重
    if_save_latest=True,                 # 是否仅保存最新权重以节省磁盘空间
    if_save_every_weights=True,          # 是否在每个保存点保存小型最终模型
    save_every_epoch=4,                  # 保存频率
    gpu_numbers1Ba="0",                  # GPU编号
    pretrained_s2G="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth", # 预训练SoVITS-G模型路径
    pretrained_s2D="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth", # 预训练SoVITS-D模型路径
    if_grad_ckpt=False,                  # 是否开启梯度检查点（v3版本）
    lora_rank="32",                      # LoRA秩
    api_name="/open1Ba"
)

# 关闭SoVITS训练进程
result = client.predict(api_name="/close1Ba")
```

#### 4.4.2 GPT训练

```python
# 执行GPT训练
result = client.predict(
    batch_size=4,                      # 每个GPU的批量大小
    total_epoch=15,                    # 总训练轮数
    exp_name="xxx",                    # 实验/模型名称
    if_dpo=False,                      # 是否启用DPO训练（实验性）
    if_save_latest=True,               # 是否仅保存最新权重以节省磁盘空间
    if_save_every_weights=True,        # 是否在每个保存点保存小型最终模型
    save_every_epoch=5,                # 保存频率
    gpu_numbers="0",                   # GPU编号
    pretrained_s1="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", # 预训练GPT模型路径
    api_name="/open1Bb"
)

# 关闭GPT训练进程
result = client.predict(api_name="/close1Bb")
```

### 4.5 TTS推理

#### 4.5.1 模型权重选择

```python
# 获取可用的模型权重列表
result = client.predict(api_name="/change_choices")
# 返回两个列表：SoVITS权重列表和GPT权重列表
```

#### 4.5.2 配置TTS推理

```python
# 配置TTS推理参数
result = client.predict(
    bert_path="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large", # 预训练中文BERT模型路径
    cnhubert_base_path="GPT_SoVITS/pretrained_models/chinese-hubert-base",  # 预训练SSL模型路径
    gpu_number="0",                 # GPU编号，只能输入一个整数
    gpt_path="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt", # GPT权重路径
    sovits_path="GPT_SoVITS/pretrained_models/s2G488k.pth", # SoVITS权重路径
    batched_infer_enabled=False,    # 是否启用并行推理版本
    api_name="/change_tts_inference"
)
```

#### 4.5.3 切换模型版本

```python
# 切换模型版本
result = client.predict(
    version_="v2",  # 可选值: 'v1', 'v2', 'v3'
    api_name="/switch_version"
)
# 返回多个参数，包括预训练模型路径、权重列表、训练参数等
```

## 5. 典型使用流程

### 5.1 训练流程

1. 数据预处理：
   - 音频切片 (`/open_slice`)
   - 音频降噪 (`/open_denoise`)
   - 语音识别 (`/open_asr`)

2. 特征提取：
   - 一键格式化训练集 (`/open1abc`) 或单独执行以下步骤:
     - BERT特征提取 (`/open1a`)
     - SSL特征提取 (`/open1b`) 
     - 语义token提取 (`/open1c`)

3. 模型训练：
   - SoVITS训练 (`/open1Ba`)
   - GPT训练 (`/open1Bb`)

### 5.2 推理流程

1. 获取可用模型列表 (`/change_choices`)
2. 配置TTS推理参数 (`/change_tts_inference`)
3. 使用另外的API进行推理（该部分API未在提供的文档中列出）

## 6. 注意事项

1. 所有路径应使用适当的格式，Windows系统中使用双反斜杠或原始字符串
2. GPU编号格式：单个GPU使用数字(如"0")，多个GPU用连字符分隔(如"0-1-2")
3. 确保所有预训练模型路径正确且可访问
4. 对于路径参数，建议使用绝对路径以避免路径解析问题
5. 大多数处理API都有对应的关闭API，使用完毕后应调用关闭API以释放资源

通过本手册中的API，可以自动化GPT-SoVITS系统的完整工作流程，从数据预处理、特征提取、模型训练到最终的语音合成。