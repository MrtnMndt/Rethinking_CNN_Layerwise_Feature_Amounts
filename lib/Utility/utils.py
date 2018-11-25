import subprocess
import torch
import shutil


class GPUMem:

    def __init__(self, is_gpu):
        self.is_gpu = is_gpu
        if self.is_gpu:
            self.total_mem = self.__get_total_gpu_memory()

    def __get_total_gpu_memory(self):
        total_mem = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.total",
                                             "--format=csv,noheader,nounits"])

        return float(total_mem[0:-1])  # gets rid of "\n" and converts string to float

    def get_mem_util(self):
        if self.is_gpu:
            # Check for memory of GPU ID 0 as this usually is the one with the heaviest use
            free_mem = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free",
                                                "--format=csv,noheader,nounits"])
            free_mem = float(free_mem[0:-1])  # gets rid of "\n" and converts string to float
            mem_util = 1 - (free_mem / self.total_mem)
        else:
            mem_util = 0

        return mem_util


def save_checkpoint(state, is_best, file_path, file_name='checkpoint.pth.tar'):
    """
    Saves the current state of the model. Does a copy of the file
    in case the model performed better than previously.

    Parameters:
        state (dict): Includes optimizer and model state dictionaries
        is_best (bool): True if model is best performing model
        file_path (str): Path to save file
        file_name (str): File name with extension (default: checkpoint.pth.tar)
    """

    save_path = file_path + '/' + file_name
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, file_path + '/model_best.pth.tar')
