"""
This file defines the paths to various resources in the project.
"""
import os
from pathlib import Path

root_path = Path(__file__).parent.parent.parent
data_path = root_path.joinpath("data")
config_path = root_path.joinpath("config")
tiny_imagenet_path = data_path.joinpath("tiny-imagenet-200")
src_path = root_path.joinpath("src")
checkpoints_path = root_path.joinpath("checkpoints")


def temp_switch_working_dir(destination):
    def decorator(f):
        def inner(*args, **kwargs):
            # Current working dir
            original_cwd = os.getcwd()
            # Change working dir to temp destination
            os.chdir(destination)
            ret = f(*args, **kwargs)
            # Change back to original working dir
            os.chdir(original_cwd)
            return ret

        return inner

    return decorator
