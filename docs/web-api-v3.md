# WebAPI文档 (3.0) - 使用了缓存技术，初始化时使用LRU Cache TTS 实例，缓存加载模型的世界，达到减少切换不同语音时的推理时间

` python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml `

## 执行参数

    `-a` - `绑定地址, 默认"127.0.0.1"`
    `-p` - `绑定端口, 默认9880`
    `-c` - `TTS配置文件路径, 默认"GPT_SoVITS/configs/tts_infer.yaml"`

## 调用

### 推理

endpoint: `/tts`
GET:

```
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

POST:

```json
{
    "text": "",                                                 # str.(required) text to be synthesized
    "text_lang": "",                                            # str.(required) language of the text to be synthesized
    "ref_audio_path": "",                                       # str.(required) reference audio path.
    "prompt_text": "",                                          # str.(optional) prompt text for the reference audio
    "prompt_lang": "",                                          # str.(required) language of the prompt text for the reference audio
    "top_k": 5,                                                 # int.(optional) top k sampling
    "top_p": 1,                                                 # float.(optional) top p sampling
    "temperature": 1,                                           # float.(optional) temperature for sampling
    "text_split_method": "cut5",                                # str.(optional) text split method, see text_segmentation_method.py for details.
    "batch_size": 1,                                            # int.(optional) batch size for inference
    "batch_threshold": 0.75,                                    # float.(optional) threshold for batch splitting.
    "split_bucket": true,                                       # bool.(optional) whether to split the batch into multiple buckets.
    "speed_factor":1.0,                                         # float.(optional) control the speed of the synthesized audio.
    "fragment_interval":0.3,                                    # float.(optional) to control the interval of the audio fragment.
    "seed": -1,                                                 # int.(optional) random seed for reproducibility.
    "media_type": "wav",                                        # str.(optional) media type of the output audio, support "wav", "raw", "ogg", "aac".
    "streaming_mode": false,                                    # bool.(optional) whether to return a streaming response.
    "parallel_infer": True,                                     # bool.(optional) whether to use parallel inference.
    "repetition_penalty": 1.35,                                 # float.(optional) repetition penalty for T2S model.
    "tts_infer_yaml_path": “GPT_SoVITS/configs/tts_infer.yaml”  # str.(optional) tts infer yaml path
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400

### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:

```
http://127.0.0.1:9880/control?command=restart
```

POST:

```json
{
    "command": "restart"
}
```

RESP: 无

### 切换GPT模型

endpoint: `/set_gpt_weights`

GET:

```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
```

RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400

### 切换Sovits模型

endpoint: `/set_sovits_weights`

GET:

```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth
```

RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400
