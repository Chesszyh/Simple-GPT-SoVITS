import torch
from torch.multiprocessing import cpu_count

def set_nvidia_gpu():
    n_cpu = cpu_count()
    
    ngpu = torch.cuda.device_count()
    gpu_infos = []
    mem = []
    if_gpu_ok = False
    
    # 判断是否有能用来训练和加速推理的N卡
    ok_gpu_keywords = {"10", "16", "20", "30", "40", "A2", "A3", "A4", "P4", "A50", "500", "A60", "70", "80", "90", "M4", "T4", "TITAN", "L4", "4060", "H", "600", "506", "507", "508", "509"}
    set_gpu_numbers = set()
    
    if torch.cuda.is_available() or ngpu != 0:
        for i in range(ngpu):
            gpu_name = torch.cuda.get_device_name(i)
            if any(value in gpu_name.upper() for value in ok_gpu_keywords):
                # A10#A100#V100#A40#P40#M40#K80#A4500
                if_gpu_ok = True  # 至少有一张能用的N卡
                gpu_infos.append("%s\t%s" % (i, gpu_name))
                set_gpu_numbers.add(i)
                mem.append(int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4))
                
    return n_cpu, gpu_infos, mem, if_gpu_ok, set_gpu_numbers

def set_default():
    global default_batch_size, default_max_batch_size, gpu_info, default_sovits_epoch
    global default_sovits_save_every_epoch, max_sovits_epoch, max_sovits_save_every_epoch
    global default_batch_size_s1, if_force_ckpt
    
    if_force_ckpt = False
    
    # GPU configuration
    if if_gpu_ok and len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
        min_memory_gb = min(mem)
        
        # Version-specific batch size calculation
        if version == "v3":
            default_batch_size = min_memory_gb // 8
        else:
            default_batch_size = min_memory_gb // 2
        
        default_batch_size_s1 = min_memory_gb // 2
    else:
        # Fallback to CPU
        gpu_info = "0\tCPU"
        gpu_infos.append("0\tCPU")
        set_gpu_numbers.add(0)
        
        # Calculate batch size based on system RAM
        system_ram_gb = int(psutil.virtual_memory().total / 1024 / 1024 / 1024)
        default_batch_size = default_batch_size_s1 = system_ram_gb // 4
    
    # Set version-specific training parameters
    if version == "v3":
        default_sovits_epoch = 2
        default_sovits_save_every_epoch = 1
        max_sovits_epoch = 3
        max_sovits_save_every_epoch = 3
    else:
        default_sovits_epoch = 8
        default_sovits_save_every_epoch = 4
        max_sovits_epoch = 25
        max_sovits_save_every_epoch = 25

    # Ensure batch sizes are at least 1
    default_batch_size = max(1, default_batch_size)
    default_batch_size_s1 = max(1, default_batch_size_s1)
    default_max_batch_size = default_batch_size * 3

set_default()

# Initialize GPU selection values
gpus = "-".join([i[0] for i in gpu_infos])
default_gpu_numbers = str(sorted(list(set_gpu_numbers))[0]) if set_gpu_numbers else "0"

def fix_gpu_number(input_gpu):
    """Ensures the GPU number is within valid range"""
    try:
        # If input is not in valid set, return default
        if int(input_gpu) not in set_gpu_numbers:
            return default_gpu_numbers
        return input_gpu
    except:
        # If input cannot be converted to int, return as is
        return input_gpu

def fix_gpu_numbers(inputs):
    """Fix multiple GPU numbers in a comma-separated string"""
    try:
        output = [str(fix_gpu_number(gpu_id)) for gpu_id in inputs.split(",")]
        return ",".join(output)
    except:
        # Return original input if processing fails
        return inputs
