from typing import Optional
import os.path

import yaml
from bravado.exception import HTTPInternalServerError
import neptune.new
from neptune.new.exceptions import MissingFieldException, NeptuneConnectionLostException
from neptune.new.internal.state import ContainerState

from utilities.path import config_path, root_path, temp_switch_working_dir


class NeptuneRun:
    stop_seconds: int = 30

    def __init__(self):
        with open(config_path.joinpath("neptune.yaml").resolve()) as reader:
            self.config = yaml.load(reader, yaml.FullLoader)
            self._is_debug_mode = self.config['mode'] == 'debug'
        self.run: Optional[NeptuneRun] = None

    def is_debug_mode(self):
        return self._is_debug_mode

    # Temporary switching working directory to change the save directory for neptune (official workaround)
    @temp_switch_working_dir(root_path)
    def __enter__(self):
        self.run = neptune.new.init_run(**self.config)
        return self

    @temp_switch_working_dir(root_path)
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.run is not None:
            self.run.stop(seconds=self.stop_seconds)

    def store(self, key, value: dict or None):
        def map_leafs_of_nested_dictionary(func, item):
            if isinstance(item, dict):
                return {k: map_leafs_of_nested_dictionary(func, item[k]) for k in item}
            return func(item)

        self.run[key] = map_leafs_of_nested_dictionary(str, value if value is not None else {})

    def log(self, key, value):
        self.run[key].log(value)

    def upload_checkpoint(self, checkpoint_full_filename):
        if self.run is None or self.run._state == ContainerState.STOPPED:
            return

        filename = os.path.basename(checkpoint_full_filename)
        key = f'model_checkpoints/{filename.replace(".pth", "")}'
        self.run[key].upload(checkpoint_full_filename)

    @staticmethod
    def handle_connection_error(function):
        def inner(*args, **kwargs):
            try:
                function(*args, **kwargs)
            except (NeptuneConnectionLostException, HTTPInternalServerError) as e:
                print(e)
                print("[fail-safe]: lost connection to neptune, gracefully storing local checkpoint")
        return inner
