### This is a set of helper functions and modules that maintain a clear log of different training runs
################################################################################################################################
from dataclasses import dataclass, asdict
import json 
import subprocess 
from datetime import datetime 
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter 

def git_info(): 
    # Function to run git commands in the command line and read their outputs:
    def run_git_command(*args):
        return subprocess.run(["git" , *args], capture_output=True , text=True).stdout.strip()

    # Get hash for the most current commit: 
    commit = run_git_command('rev-parse' , 'HEAD')

    # Check whether the training log is running code that has already been tracked by git:
    not_tracked = bool(run_git_command('status' , '--porcelain'))

    # If not_tracked == False, then the code being run has already been committed to the repository. If not, we should append all the undocumented changes uses git diff: 
    if not_tracked == True: 
        # Document differences: 
        diffs = run_git_command('diff' , 'HEAD')
        return {'commit': commit , 'not_tracked' : not_tracked , 'uncommitted' : diffs}
    else:
        return {'commit': commit , 'not_tracked' : not_tracked }



def log_run(config, root):
    # Function to log runs along with their git info: 

    # Date and time stamp: 
    date_time_stamp = datetime.now().strftime("%d%m%Y-%H%M%S")
    run_directory = Path(root)/f"{date_time_stamp}_lr{config.LEARNING_RATE}_gamma{config.GAMMA}_seed{config.SEED}"

    # make the directory:
    run_directory.mkdir(parents = True)

    # Metadata for the run: 
    metadata = {'config': asdict(config) , 'git_info' : git_info()}
    (run_directory / "metadata.json").write_text(json.dumps(metadata , indent=2 , default=str))
    
    # Writing the tensorboard logs: 
    writer = SummaryWriter(run_directory)
    writer.add_text("metadata", "```\n" + json.dumps(metadata, indent=2, default=str) + "\n```")
    return run_directory , writer