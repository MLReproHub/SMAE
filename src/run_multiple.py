from enum import Enum
import os
import os.path
import yaml
import copy
import subprocess

from typing import MutableMapping
from sympy import parse_expr

Phase = Enum('Phase', ('PRETRAIN', 'DOWNSTREAM'))

current_phase = Phase.DOWNSTREAM

Config = Enum('Config', ('MODEL', 'TRAIN'))

model_config_dir = "config/model/"
train_config_dir = "config/train/"

model_key, model_config = "mae", "e7d2_128"

train_key, train_config = "pretrain", "gridsearch"
downstream_train_key, downstream_train_config = "linearprobing", "default"

base_model_filename = model_config_dir + model_key + '_' + model_config
base_train_filename = train_config_dir + train_key + '_' + train_config


def main():
    variants = (
        ((Config.TRAIN, 'optim.AdamW.lr', 1.5e-4), (Config.TRAIN, 'optim.AdamW.weight_decay', 0.15), (Config.MODEL, 'MAE.decoder.TransformerDecoder.num_layers', 2)),
        ((Config.TRAIN, 'optim.AdamW.lr', 1.5e-4), (Config.TRAIN, 'optim.AdamW.weight_decay', 0.05), (Config.MODEL, 'MAE.decoder.TransformerDecoder.num_layers', 3)),
        ((Config.TRAIN, 'optim.AdamW.lr', 1.5e-4), (Config.TRAIN, 'optim.AdamW.weight_decay', 0.15), (Config.MODEL, 'MAE.decoder.TransformerDecoder.num_layers', 3)),
        ((Config.TRAIN, 'optim.AdamW.lr', 1.5e-4), (Config.TRAIN, 'optim.AdamW.weight_decay', 0.05), (Config.MODEL, 'MAE.decoder.TransformerDecoder.num_layers', 2)),
    )

    run_variants(variants)


def run_variants(variants):
    base_model_config = load(base_model_filename)
    base_train_config = load(base_train_filename)

    for index, variant in enumerate(variants):
        if current_phase == Phase.PRETRAIN:
            run_pretrain_variant(
                index,
                variant,
                copy.deepcopy(base_model_config),
                copy.deepcopy(base_train_config))
        elif current_phase == Phase.DOWNSTREAM:
            run_downstream_variant(index)


def load(filename):
    full_filename = filename + '.yaml'
    assert os.path.exists(full_filename), full_filename
    file = open(full_filename)
    return yaml.load(file, ConfigFullLoader)


class ConfigFullLoader(yaml.FullLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self.add_constructor(tag='!eval', constructor=self.evaluate)

    @staticmethod
    def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.') -> MutableMapping:
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(ConfigFullLoader.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def evaluate(loader: yaml.Loader, node: yaml.nodes.MappingNode):
        # noinspection PyTypeChecker
        expr = loader.construct_scalar(node)
        try:
            fd = ConfigFullLoader.flatten_dict(list(loader.constructed_objects.values())[0])
            val = round(float(parse_expr(expr, local_dict={k.split('.')[-1]: v for k, v in fd.items()})), 6)
            if val == int(val):
                val = int(val)
        except (ValueError, TypeError):
            return expr
        return val



def set_value(base_config, key, value):
    keys = key.split('.')
    all_but_last_key = keys[:-1]
    last_key = keys[-1]

    lookup = base_config

    for key in all_but_last_key:
        lookup = lookup[key]

    lookup[last_key] = value


def run_pretrain_variant(index, variant, base_model_config, base_train_config):

    for config, key, value in variant:
        base_config = None

        if config == Config.MODEL:
            base_config = base_model_config

        elif config == Config.TRAIN:
            base_config = base_train_config

        set_value(base_config, key, value)

    def write(filename, index, config):
        file = open(filename + '-' + str(index) + '.yaml', 'w')
        yaml.dump(config, file)

    write(base_model_filename, index , base_model_config)
    write(base_train_filename, index, base_train_config)

    args = (
        "python",
        "src/main.py",
        f"--model_key={model_key}",
        f"--model_config={model_config}-{index}",
        f"--intention={train_key}",
        f"--train_config={train_config}-{index}")

    print(f"Running {args}")

    subprocess.run(args)

    print(f"Done running {args}")

def run_downstream_variant(index):

    args = (
        "python",
        "src/main.py",
        "--resume",
        f"--model_key={model_key}",
        f"--model_config={model_config}-{index}",
        f"--intention={downstream_train_key}",
        f"--train_config={downstream_train_config}")

    print(f"Running {args}")

    subprocess.run(args)

    print(f"Done running {args}")

if __name__ == "__main__":
    main()
