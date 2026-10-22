### This is a set of helper functions and modules that maintain a clear log of different training runs
################################################################################################################################
from dataclasses import dataclass
import json 
import subprocess 
from datetime import datetime 
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter 

def git_info(): 
    def run(*args):
        return subprocess.run(["git" , *args], capture_output=True , text=True).stdout.strip()
    return {'commit': run('rev-parse' , 'HEAD') , }
