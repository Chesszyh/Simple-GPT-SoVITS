import os
import sys
import signal
from sys import argv # argv: 命令行参数列表

# 处理命令行命令 NOTE 为啥有这个函数？
def handle_control(command:str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
        
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