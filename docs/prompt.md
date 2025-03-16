# Copilot

## 重构代码

请你阅读并重构以下项目的核心代码文件 `api.py`）。在重构中需要：

1. **保留所有原本功能**  
   - 不要丢失任何关键变量、函数调用以及流程步骤。  
   - 原先用到的外部脚本 (`GPT_SoVITS`、`tools` 等) 依旧通过子进程或函数调用的方式进行管理。

2. **改进代码结构与可读性**  
   - 按照逻辑将功能拆分到更合适的函数或类中；  
   - 使用清晰、统一的命名；  
   - 去除多余的全局变量和环境变量写法；  
   - 提高异常处理与日志输出的可读性。

3. **提高可维护性**  
   - 在必要处增加注释和文档字符串，解释函数或类的用途；  
   - 对主要函数添加类型注解（type hints）；  
   - 将与系统配置相关的逻辑（如 GPU 检测、默认路径等）集中在 `Config` 类或配置文件统一管理；  
   - 保持同一处只处理单一责任，减少交叉引用或频繁更新全局状态。

4. **改进健壮性**  
   - 针对可能出错的流程（如文件路径、子进程启动）增加必要的错误检查或重试逻辑；  
   - 用更明确的条件判断代替奇怪的 if/else 嵌套结构；  
   - 保证子进程启动和停止时有清晰的管理，避免僵尸进程。

5. **尽量不改变文件接口**  
   - 外部调用方式（如 `python api.py --xxx`）保持兼容；  
   - WebUI、FastAPI 接口路由、请求参数、返回格式保持一致，除非实在需要做小幅改动来配合重构。

6. **可选改进**  
   - 如果脚本中有大量冗余或重复片段，可简化并合并；  
   - 可考虑引入更好的日志系统或配置加载机制；  
   - 如有启用多 GPU、分布式等需求，可统一封装到一个并行管理模块里。

请在保证功能不变的前提下，为`api.py`这个文件生成重构后的代码，并在需要处添加必要说明。  

---

执行以上指令后，重构工具将输出新的代码版本。请务必在输出的文件中保留最核心的功能调用。

## Gradio -> API

"Convert the following Gradio-based WebUI code into a FastAPI (or other API-based) implementation to facilitate AI Agent calls. The API should preserve all functionalities, including user inputs, file processing, and interactions between components. The endpoints should be structured in a RESTful way, ensuring easy integration. Additionally, replace Gradio UI components (like `Textbox`, `Button`, `Dropdown`) with appropriate API endpoints that can handle equivalent operations. Return JSON responses where applicable. The API should also allow batch processing and handle concurrent requests efficiently.  

Ensure that the final API implementation is well-structured, modular, and scalable for future extensions.

NOTICE:DO NOT OMIT ANY PART OF THE CODE, EVEN IF THEY HAVE QUITE SIMILAR LOGIC!!! YOU MUST OUTPUT YOUR COMPLETE CODE!!!"


Here's the full Gradio code:  

```python
# (Insert the entire Gradio code here)
```

### NOTE

我发现gradio好像自带API调用方式，所以我做的好像又有点多此一举了。
