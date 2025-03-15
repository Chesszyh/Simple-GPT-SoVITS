# 探索除GPT-Sovits以外的TTS解决方案

## [ChatTTS](https://github.com/2noise/ChatTTS/)

### Issue

- 配置python环境(具体而言，安装pynini--处理加权有限状态转换器的包)时报错：`ModuleNotFoundError: No module named 'Cython' [end of output]`
  - 安装Cython后再次尝试pip install，还是报错：`fatal error: fst/util.h: No such file or directory`
    - 尝试：`apt update` + `apt install openfst-dev`，但是apt无法找到 `openfst-dev` 软件包
      - `search openfst`查找可用包，无结果
      - 从源代码编译openfst：
        - 下载
        - 解压：`tar -zxvf openfst-x.x.x.tar.gz`
          - 后续安装还是有问题
  - 解决：直接`pip install chatTTS`

- 使用时报错：可能是模型不兼容的问题？
  - 多家TTS项目的模型，彼此之间可能是不兼容的，除非懂得底层原理能够自行调整

## [Fish-speech](https://github.com/fishaudio/fish-speech)

[Doc](https://docs.fish.audio/introduction)

文档不全面

## [F5-TTS](https://github.com/SWivid/F5-TTS)

这个包含在了GPT-Sovits中，但不知道效果如何