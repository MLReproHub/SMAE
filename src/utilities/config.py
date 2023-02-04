import abc
import copy
import importlib
import importlib.util
import inspect
import itertools
import os.path
import pathlib
import pkgutil
import re
import sys
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from functools import partial
from inspect import Parameter
from types import ModuleType
from typing import Type, Callable, Dict, List, Optional, OrderedDict as OrderedDictT, TextIO, Tuple, NamedTuple, Any, \
    MutableMapping, Generator, Union

import torch
import yaml
from sympy import parse_expr
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader

from loss.uncertainty import UncertaintyWeightedLoss
from utilities.neptune import NeptuneRun
from utilities.path import config_path, src_path, checkpoints_path, root_path
from utilities.train import TrainingSetup


class IArgResolver(metaclass=abc.ABCMeta):
    CACHE: Dict[str, Dict[str, str]] = {}

    @abc.abstractmethod
    def resolve(self, class_name: str, instance: nn.Module) -> str or None:
        raise NotImplementedError

    def cache(self, cls_name: str, arg: str, arg_cls_name: str) -> None:
        if cls_name not in self.__class__.CACHE.keys():
            self.__class__.CACHE[cls_name] = {}
        if arg not in self.__class__.CACHE[cls_name].keys():
            self.__class__.CACHE[cls_name][arg] = arg_cls_name

    @abc.abstractmethod
    def state_dict_params(self) -> dict:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {
            'class': self.__class__.__name__,
            'init_kwargs': self.state_dict_params()
        }

    @staticmethod
    def from_sd(sd: dict) -> 'IArgResolver':
        cls_name = sd['class']
        cls = None
        for sc in IArgResolver.__subclasses__():
            if sc.__name__ == cls_name:
                cls = sc
                break
        assert cls is not None, f'Subclass named "{cls_name}" is not a subclass of IArgResolver'
        # noinspection PyArgumentList
        return cls(**sd['init_kwargs'])


class IClassResolver(metaclass=abc.ABCMeta):
    CACHE: Dict[str, Type[nn.Module]] = {}
    CONDE_ENV_PATH: str = pathlib.Path(os.sep.join(sys.exec_prefix.split(os.sep)))

    def cache(self, class_name: str, cls: Type[nn.Module] or Callable[[str], Type[nn.Module]]):
        if class_name not in self.__class__.CACHE.keys():
            if callable(cls):
                cls = cls(class_name)
            self.__class__.CACHE[class_name] = cls
        else:
            print(f'\t[IClassResolver::cache] Cache hit for class "{class_name}"')

    def can_resolve(self, class_name) -> bool:
        try:
            return self.resolve(class_name) is not None
        except:
            pass
        return False

    @abc.abstractmethod
    def resolve(self, class_name: str) -> nn.Module or None:
        raise NotImplementedError

    @abc.abstractmethod
    def state_dict_params(self) -> dict:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {
            'class': self.__class__.__name__,
            'init_kwargs': self.state_dict_params()
        }

    @staticmethod
    def fix_abs_path(p: str) -> str:
        project_name = root_path.stem
        p_parts = p.split('/' if p.startswith('/') else '\\')
        return os.path.join(str(root_path), *p_parts[p_parts.index(project_name) + 1:])

    @staticmethod
    def fix_conda_path(p: str) -> str or None:
        p_parts = p.split('/' if p.startswith('/') else '\\')
        new_path = os.path.join(str(IClassResolver.CONDE_ENV_PATH), *p_parts[p_parts.index('envs') + 2:])
        return new_path if os.path.exists(new_path) else None

    @staticmethod
    def fix_abs_path_rec(d: dict) -> dict:
        for k in d.keys():
            v = d[k]
            if type(v) == str:
                if 'conda' in v.lower():
                    d[k] = IClassResolver.fix_conda_path(v)
                elif v.startswith('/') or ':\\' in v.lower():
                    d[k] = IClassResolver.fix_abs_path(v)
        return d

    @staticmethod
    def from_sd(sd: dict) -> 'IClassResolver':
        cls_name = sd['class']
        cls = None
        for sc in IClassResolver.__subclasses__():
            if sc.__name__ == cls_name:
                cls = sc
                break
        assert cls is not None, f'Subclass named "{cls_name}" is not a subclass of IClassResolver'
        init_kwargs = IClassResolver.fix_abs_path_rec(sd['init_kwargs'])
        # noinspection PyArgumentList
        return cls(**init_kwargs)


# ==================================


class ConfigArgResolver(IArgResolver):
    def __init__(self, mapping: Dict[str, Type[nn.Module] or str] or None = None):
        self.__class__.CACHE = {}
        if mapping is None:
            mapping = {}
        for cls_dot_arg, arg_cls_name in mapping.items():
            cls_name, arg_name = cls_dot_arg.split('.', maxsplit=1)
            if type(arg_cls_name) != str:
                arg_cls_name = arg_cls_name.__name__ if hasattr(arg_cls_name, '__name__') else str(arg_cls_name)
            self.cache(cls_name, arg_name, arg_cls_name)
        self.mapping = mapping

    def resolve(self, class_name: str, instance: nn.Module) -> str or None:
        if class_name not in self.__class__.CACHE.keys():
            return None
        for arg_name, arg_class_name in self.__class__.CACHE[class_name].items():
            if arg_class_name == instance.__class__.__name__ or (
                    arg_class_name.startswith('*') and
                    instance.__class__.__name__.endswith(arg_class_name.replace('*', ''))
            ):
                return arg_name
        return None

    def state_dict_params(self):
        return {'mapping': self.mapping}


class GlobalClassResolver(IClassResolver):
    def resolve(self, class_name: str) -> Type[nn.Module] or None:
        try:
            self.cache(class_name, lambda cn: globals()[cn])
            return IClassResolver.CACHE[class_name]
        except KeyError:
            print(f'[GlobalClassResolver::resolve] Class resolution FAILed (class_name="{class_name}")',
                  file=sys.stderr)
            return None

    def state_dict_params(self):
        return {}


class ImportedModuleClassResolver(IClassResolver):
    # noinspection PyBroadException
    def __init__(self, module_name: Optional[str] = None, module_file: Optional[str] = None,
                 module: Optional[ModuleType] = None):
        if module is None and module_file is None:
            # all hopes go to import_module("module_name")
            module = importlib.import_module(module_name)
        if module is not None:
            module_name, module_file = module.__name__, module.__file__
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        self.module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(self.module)
        except:
            pass
        self.module_name, self.module_file = module_name, module_file

    def resolve(self, class_name: str) -> Type[nn.Module] or None:
        if hasattr(self.module, class_name):
            return getattr(self.module, class_name)
        return None

    def state_dict_params(self):
        return {'module_name': self.module_name, 'module_file': self.module_file}


class RecursiveModuleClassResolver(IClassResolver):
    def __init__(self, module_name: str = os.path.join(src_path, 'model')):
        self.module_name = module_name
        self.modules = [importlib.import_module('.' + name, os.path.basename(module_name))
                        for (_, name, _) in pkgutil.iter_modules([module_name])]

    def resolve(self, class_name: str) -> Type[nn.Module] or None:
        for module in self.modules:
            if hasattr(module, class_name):
                return getattr(module, class_name)
        return None

    def state_dict_params(self):
        return {'module_name': self.module_name}


# ==================================

@dataclass
class ConfigEntry:
    class_name: str
    cls: Type[nn.Module]
    init_kwargs: dict = field(default_factory=dict)
    instance: Optional[nn.Module] = None
    overridden_kwargs: Optional[dict] = None

    @property
    def cls_args(self) -> Dict[str, Parameter]:
        # noinspection PyTypeChecker
        return inspect.signature(getattr(self.cls, '__init__')).parameters

    @property
    def original_args(self) -> Dict[str, int or float or str or bool]:
        return {k: v for k, v in self.init_kwargs.items() if not hasattr(v, '__dict__')}

    @property
    def resolved(self) -> bool:
        return self.instance is not None

    @property
    def unresolved_args(self) -> List[str]:
        return [p.name for p in self.cls_args.values()
                if inspect.isclass(p.annotation) and issubclass(p.annotation, nn.Module)
                and p.name not in self.init_kwargs.keys()]

    def clone(self) -> 'ConfigEntry':
        return ConfigEntry(self.class_name, self.cls, self.init_kwargs, self.instance)

    def collect_overridden(self, od: dict) -> Optional[Dict[str, Any]]:
        found_k = None
        for k in od.keys():
            if re.search(k, self.class_name):
                found_k = k
                break
        collected = None
        if found_k is not None:
            self.overridden_kwargs = od.pop(found_k)
            collected = {found_k: copy.deepcopy(self.overridden_kwargs)}
            self._process_overridden_args()
            if self.overridden_kwargs is not None and len(self.overridden_kwargs):
                print(f'\t[ConfigEntry::collect_overridden] collecting "{found_k}": {str(self.overridden_kwargs)}')
        return collected

    def _recursively_resolved(self, d: dict or Any) -> bool:
        if type(d) != dict:
            if type(d) == ConfigEntry:
                return d.resolved
            return True
        for k, v in d.items():
            if k.startswith('__'):
                continue
            if type(v) == ConfigEntry:
                if not v.resolved:
                    return False
                d[k] = v.instance
            if type(v) == dict and not self._recursively_resolved(v):
                return False
            elif type(v) == list:
                for i, vi in enumerate(v):
                    if not self._recursively_resolved(vi):
                        return False
                    elif type(vi) == ConfigEntry:
                        d[k][i] = vi.instance
        return True

    def resolvable(self) -> bool:
        return (self.unresolved_args is None or len(self.unresolved_args) == 0) \
               and self._recursively_resolved(self.init_kwargs)

    def _parse_deferred_args(self, deferred_args: dict or None) -> dict:
        if deferred_args is None or type(deferred_args) != dict:
            return {}
        for dk in deferred_args.keys():
            if dk in self.cls_args.keys() and dk not in self.init_kwargs.keys():
                self.init_kwargs[dk] = deferred_args[dk]

    def _process_overridden_args(self):
        if self.overridden_kwargs is not None and type(self.overridden_kwargs) == dict:
            for dk in list(self.overridden_kwargs.keys()):
                if dk in self.cls_args.keys() and dk in self.init_kwargs.keys():
                    if type(self.overridden_kwargs[dk]) == dict:
                        child_ce = self.init_kwargs[dk]
                        if type(child_ce) != ConfigEntry:
                            child_ce = child_ce[list(child_ce.keys())[-1]]
                        assert isinstance(child_ce, ConfigEntry)
                        child_ce.collect_overridden({str(child_ce.class_name): self.overridden_kwargs.pop(dk)})

    def _override_init_kwargs(self):
        if self.overridden_kwargs is not None:
            for k, v in self.overridden_kwargs.items():
                if k in self.init_kwargs.keys():
                    self.init_kwargs[k] = v

    def _process_init_kwargs(self):
        if self.init_kwargs is None:
            self.init_kwargs = {}
        self.init_kwargs = {k: v for k, v in self.init_kwargs.items() if not k.startswith('__')}
        for k in self.init_kwargs.keys():
            v = self.init_kwargs[k]
            if type(v) == dict and len(v.keys()) in [1, 2]:
                for vk, vv in v.items():
                    if vk == '__resolved':
                        continue
                    if type(vv) == ConfigEntry and vk == vv.class_name and vv.resolved:
                        self.init_kwargs[k] = vv.instance
                        break
                    if isinstance(vv, object) and vk == vv.__class__.__name__:
                        self.init_kwargs[k] = vv
                        break

    # noinspection PyArgumentList,PyTypeChecker,PyBroadException
    def resolve(self, deferred_args=None):
        # noinspection PyArgumentList
        self._process_init_kwargs()
        self._override_init_kwargs()
        self._parse_deferred_args(deferred_args)
        try:
            self.instance = self.cls(**self.init_kwargs)
        except Exception as e:
            print(f'\t[ConfigEntry::resolve] {str(e)}', file=sys.stderr)
            print(f'\t[ConfigEntry::resolve] Initializing {self.class_name} as partial with args: {self.init_kwargs}')
            self.instance = partial(self.cls, **self.init_kwargs)
        return self.instance

    @staticmethod
    def from_cloned(ce: 'ConfigEntry') -> 'ConfigEntry':
        cn = ConfigEntry('', nn.Module)
        cn.class_name = ce.class_name
        cn.cls = ce.cls
        cn.init_kwargs = ce.init_kwargs
        cn.instance = ce.instance
        return cn


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


CLAR = Tuple[nn.Module, TrainingSetup, Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], 'ConfigReader']
BPD = Tuple[List[ConfigEntry], int, List[Tuple[str, int or float or bool or dict]], dict, dict, str]


class ConfigReader:
    INSTANCES: List['ConfigReader'] = []

    def __init__(self, cls_resolvers: List[IClassResolver] or IClassResolver or None = None,
                 arg_resolver: IArgResolver or None = None, override_keys: dict or None = None, chkpt_fpath=None):
        if cls_resolvers is None:
            self.cls_resolvers = [GlobalClassResolver()]
        else:
            self.cls_resolvers = cls_resolvers if type(cls_resolvers) is list else [cls_resolvers]
        self.arg_resolver = ConfigArgResolver(mapping=dict()) if arg_resolver is None else arg_resolver
        self.override_keys = {} if override_keys is None else ConfigReader._to_nested_dict(override_keys)
        self.entries: List[ConfigEntry] = []
        self.n_unresolved = 0
        self.unused = []
        self._yaml_dict = None  # model config
        self.train_config = None  # train config
        self.chkpt_fpath = chkpt_fpath
        if len(ConfigReader.INSTANCES) > 0:
            warnings.warn('\t[ConfigReader::__init__] trying to re-instantiate class object (should be singleton)')
        ConfigReader.INSTANCES.append(self)

    @property
    def unused_dict(self) -> OrderedDictT:
        return OrderedDict(self.unused)

    @property
    def yaml_dict(self) -> dict or OrderedDictT:
        return self._yaml_dict

    @yaml_dict.setter
    def yaml_dict(self, yd: dict or OrderedDictT) -> None:
        self.entries: List[ConfigEntry] = []
        self.n_unresolved = 0
        self.unused = []
        self._yaml_dict = yd

    def capture(self) -> BPD:
        return [e.clone() for e in self.entries], self.n_unresolved, copy.deepcopy(self.unused), \
               copy.deepcopy(self._yaml_dict), self.train_config, self.chkpt_fpath

    def get_config_of_instance(self, instance: Any) -> dict or None:
        for entry in self.entries:
            if entry.instance is not None and entry.instance == instance:
                return entry.original_args
        return None

    @staticmethod
    def _replace_in_dict(inp: dict, key: str, value: dict or Any) -> dict:
        if type(value) == dict and re.search(list(value.keys())[0], key):
            value = value[list(value.keys())[0]]
        if key not in inp.keys():
            for key_, value_ in inp.items():
                if type(value_) == dict and key in value_.keys():
                    inp = inp[key_]
                    break
            else:
                return inp
        for k, v in value.items():
            if k in inp[key].keys():
                if type(v) == dict and type(inp[key]) == dict:
                    inp[key] = ConfigReader._replace_in_dict(inp[key], k, v)
                else:
                    inp[key][k] = v
            elif type(inp[key][list(inp[key].keys())[0]]) == dict and k in inp[key][list(inp[key].keys())[0]].keys():
                inp[key][list(inp[key].keys())[0]][k] = v
            else:
                raise AttributeError(f'\t[ConfigReader::_replace_in_dict] {key}-{value}: Not found in dict. ({inp})')
        return inp

    def _load_dict(self, d: dict or Any, depth: int = 0, unpack: bool = False) -> dict or Any:
        if type(d) == dict and '__resolved' in d.keys() and d['__resolved']:
            for dk, dv in d.items():
                if type(dv) == ConfigEntry:
                    dv.instance = None  # force re-resolution
                    self.entries.append(dv)
            return d
        elif type(d) != dict:
            if type(d) == str:
                # Class Resolution
                for resolver in self.cls_resolvers:
                    if resolver.can_resolve(d):
                        cls = resolver.resolve(class_name=d)
                        ce = ConfigEntry(class_name=d, cls=cls, init_kwargs={})
                        self.entries.append(ce)
                        return ce
            return d

        d_out = {'__resolved': True} if not unpack else []
        for k, v in d.items():
            # Recursive Resolution
            if type(v) == list:
                v = [self._load_dict(vi, depth + 1, unpack=True) for vi in v]
            else:
                v = self._load_dict(v, depth + 1)

            # Class Resolution
            for resolver in self.cls_resolvers:
                if resolver.can_resolve(k):
                    cls = resolver.resolve(class_name=k)
                    break
            else:
                if not unpack:
                    d_out[k] = v
                else:
                    d_out.append(v)
                if depth == 0:
                    self.unused.append((k, v))
                continue
            ce = ConfigEntry(class_name=k, cls=cls, init_kwargs=v)
            if self.override_keys is not None:
                collected = ce.collect_overridden(self.override_keys)
                if collected is not None:
                    ConfigReader._replace_in_dict(self._yaml_dict, k, collected)
                    if self.train_config is not None:
                        ConfigReader._replace_in_dict(self.train_config, k, collected)
            self.entries.append(ce)
            if not unpack:
                d_out[k] = ce
            else:
                d_out.append(ce)
        if unpack and len(d_out) == 1:
            return d_out[0]
        return d_out

    def load(self, yaml_fp_or_stream_or_dict: Union[str, dict, TextIO]) -> None:
        if type(yaml_fp_or_stream_or_dict) in [dict, OrderedDict, NamedTuple]:
            yaml_dict = yaml_fp_or_stream_or_dict
        else:
            yaml_dict = yaml.load(yaml_fp_or_stream_or_dict, Loader=ConfigFullLoader)
        try:
            self.yaml_dict = copy.deepcopy(yaml_dict)
        except TypeError:
            self.yaml_dict = yaml_dict

        self._load_dict(self.yaml_dict, depth=0)
        self.n_unresolved = len(self.entries)

    def resolved(self, resolved_entry: ConfigEntry):
        print(f'\t[ConfigReader::resolved] module resolved'
              f'{" (partially)" if type(resolved_entry.instance) == partial else ""}'
              f': {resolved_entry.class_name}')
        for entry in [e for e in self.entries if not e.resolved and len(e.unresolved_args) > 0]:
            entry_arg = self.arg_resolver.resolve(entry.class_name, resolved_entry.instance)
            if entry_arg is not None:
                entry.init_kwargs[entry_arg] = resolved_entry.instance
        self.n_unresolved -= 1

    def resolve(self, only_keys=None, deferred_args=None) -> OrderedDictT[str, nn.Module]:
        n_iters, instances = 0, []
        while self.n_unresolved > 0 and n_iters < 10:
            for entry_i, entry in enumerate(self.entries):
                if entry.resolvable() and not entry.resolved:
                    entry.resolve(deferred_args=deferred_args)
                    self.resolved(entry)
                    if only_keys is None or entry.class_name in only_keys or entry.class_name.lower() in only_keys:
                        instances.append((entry.class_name, entry.instance))
            n_iters += 1
        return ConfigReader._to_ordered_dict(instances)

    def resolve_singleton(self, config: dict or OrderedDictT, return_top: bool = False, **resolve_kwargs):
        backup = self.capture()
        cls_name = [k for k in config.keys() if not k.startswith('__')][0]
        # config[cls_name] = {**config[cls_name], **resolve_kwargs.pop('deferred_args', {})}
        self.load(config)
        resolved = self.resolve(**resolve_kwargs)
        if cls_name in resolved.keys() or return_top:
            if cls_name not in resolved.keys():
                cls_name = list(resolved.keys())[0]
            resolved = resolved[cls_name]
        self.restore(backup)
        return resolved

    def restore(self, data: BPD) -> None:
        self.entries = [ConfigEntry.from_cloned(e) for e in data[0]]
        self.n_unresolved = data[1]
        self.unused = copy.deepcopy(data[2])
        self._yaml_dict = copy.deepcopy(data[3])
        # self.train_config = copy.deepcopy(data[4]) DO NOT RESTORE TRAIN CONFIG WHEN DOING GRID SEARCH
        self.chkpt_fpath = data[5]

    def state_dict(self) -> dict:
        return {
            'config': self.yaml_dict,
            'train_config': self.train_config,
            'cls_resolver': [cr.state_dict() for cr in self.cls_resolvers],
            'arg_resolver': self.arg_resolver.state_dict()
        }

    # ---------------------------------

    # def

    # ---------------------------------

    @staticmethod
    def _to_ordered_dict(items: list) -> OrderedDictT:
        items_uk, items_uv = [], []  # unique keys and corresponding merged values
        for item in items:
            item_key, item_value = item
            if item_key not in items_uk:
                items_uk.append(item_key)
                items_uv.append(item_value)
            else:
                uv_item = items_uv[items_uk.index(item_key)]
                if type(uv_item) == list:
                    uv_item.append(item_value)
                else:
                    items_uv[items_uk.index(item_key)] = [uv_item, item_value]
        return OrderedDict(zip(items_uk, items_uv))

    @staticmethod
    def _to_nested_dict(orig_dict: dict) -> dict:
        def deep_dict():
            return defaultdict(deep_dict)

        result = deep_dict()

        def deep_insert(key, value):
            d = result
            keys = key.split(".")
            for subkey in keys[:-1]:
                d = d[subkey]
            d[keys[-1]] = value

        for orig_dict_k, orig_dict_v in orig_dict.items():
            deep_insert(orig_dict_k, orig_dict_v)

        def cast_to_dict(input_dict: dict or defaultdict):
            for k in input_dict.keys():
                if type(input_dict[k]) == defaultdict:
                    input_dict[k] = cast_to_dict(input_dict[k])
            return dict(input_dict)

        return cast_to_dict(result)

    @staticmethod
    def from_sd(sd: dict) -> 'ConfigReader':
        cr = ConfigReader()
        cr.cls_resolvers = [
            *[RecursiveModuleClassResolver(os.path.join(src_path, module_name))
              for module_name in ['dataset', 'loss', 'model', ]],
            ImportedModuleClassResolver(module=torch.nn),
            ImportedModuleClassResolver(module=torch.nn.init),
            ImportedModuleClassResolver(module=torch.optim),
            ImportedModuleClassResolver(module=torch.optim.lr_scheduler),
        ]
        cr.arg_resolver = IArgResolver.from_sd(sd['arg_resolver'])
        cr.load(sd['config'])
        cr.train_config = sd.pop('train_config', None)
        return cr

    @staticmethod
    def from_config(cfg: Union[str, dict, TextIO, None] = None, cr: Optional['ConfigReader'] = None,
                    deferred_args: dict or None = None, args_mapping: dict or None = None,
                    **cr_kwargs) -> Tuple[OrderedDictT[str, Module], 'ConfigReader']:
        if cr is None:
            cr = ConfigReader(
                cls_resolvers=[
                    *[RecursiveModuleClassResolver(os.path.join(src_path, module_name))
                      for module_name in ['dataset', 'loss', 'model', ]],
                    ImportedModuleClassResolver(module=torch.nn),
                    ImportedModuleClassResolver(module=torch.nn.init),
                    ImportedModuleClassResolver(module=torch.optim),
                    ImportedModuleClassResolver(module=torch.optim.lr_scheduler),
                ],
                arg_resolver=ConfigArgResolver(args_mapping),
                **cr_kwargs
            )
            cr.load(cfg)
        else:
            assert isinstance(cr, ConfigReader)
            if 'override_keys' in cr_kwargs is not None:
                cr.override_keys = cr_kwargs['override_keys']
        resolved = cr.resolve(deferred_args=deferred_args)
        return resolved, cr

    @staticmethod
    def load_model(model_key: str = None, model_config: str = 'default', args_mapping: dict or None = None,
                   cr: 'ConfigReader' or None = None, override_keys: dict or None = None,
                   **unused_kwargs) -> Tuple[nn.Module, 'ConfigReader']:
        assert model_key is not None or cr is not None, 'Either model_key or cr must be provided.'
        cfg, chkpt_fpath = None, None
        if model_key is not None:
            yaml_fpath = os.path.join(config_path, 'model', (model_key + '_' + model_config).lower() + '.yaml')
            assert os.path.exists(yaml_fpath), yaml_fpath
            cfg = open(yaml_fpath)
            chkpt_fpath = f'{model_key}_{model_config}.pth'.lower()
        resolved, cr = ConfigReader.from_config(
            cfg,
            cr=cr,
            args_mapping=args_mapping,
            override_keys=override_keys,
            chkpt_fpath=chkpt_fpath,
        )
        cr.unused += [(k, v) for k, v in unused_kwargs.items()]
        return resolved.popitem()[-1], cr

    @staticmethod
    def load_all_grid_search(grid: Dict[str, list], load_kwargs):
        def _to_iterable(obj: Any):
            return obj if type(obj) in [list, tuple] else [obj]

        gvs = [_to_iterable(v) for v in grid.values()]
        for grid_values in itertools.product(*gvs):
            hparams_dict = dict(zip(grid.keys(), grid_values))
            load_kwargs['override_keys'] = hparams_dict
            hp_indices = tuple([hpv.index(v) for hpv, v in zip(gvs, hparams_dict.values())])
            yield ConfigReader.load_all(**load_kwargs), hparams_dict, hp_indices

    @staticmethod
    def load_all(gs_dict: Optional[Dict[str, list]] = None, **load_kwargs) -> CLAR or Generator[CLAR, None, None]:
        if gs_dict is not None:
            return ConfigReader.load_all_grid_search(gs_dict, load_kwargs)

        device = load_kwargs.pop('device', 'cpu')
        # Instantiate Model
        model, cr = ConfigReader.load_model(**load_kwargs)
        unused_keys = copy.deepcopy(cr.unused_dict)
        model = model.to(device)

        # Load train config
        train_config_arg = load_kwargs.pop('train_config', None)
        if cr.train_config is not None and len(cr.train_config.keys()) > 0 and train_config_arg is None:
            # train_config = cr.train_config
            ...
        else:
            if train_config_arg is None:
                train_config_arg = ''
            train_config_fpath = os.path.join(str(config_path), 'train', f'{train_config_arg}.yaml')
            if os.path.exists(train_config_fpath) and os.path.isfile(train_config_fpath):
                with open(train_config_fpath) as tyfp:
                    train_config = yaml.load(tyfp, Loader=ConfigFullLoader)
            else:
                train_config = {}
            cr.train_config = copy.deepcopy(train_config)
        if sys.version_info[0] >= 3 and sys.version_info[1] >= 9:
            unused_keys |= cr.train_config
        else:
            unused_keys = {**unused_keys, **cr.train_config}

        # Instantiate TrainingSetup
        ts = TrainingSetup(num_epochs=unused_keys.pop('num_epochs', 1), seed=unused_keys.pop('seed', 0),
                           device=device, train_config=cr.train_config)
        for key in ['init_fn', 'optim', 'scheduler', 'loss']:
            if key in unused_keys.keys():
                key_dict = unused_keys.pop(key)
                deferred_args = {
                    'optim': {'params': model.parameters()},
                    'scheduler': {'optimizer': ts.optimizer},
                }.get(key, {})
                cls = cr.resolve_singleton(key_dict, deferred_args=deferred_args)

                if 'init_fn' == key:
                    def weight_initializer(m: nn.Module):
                        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
                            cls(m.weight)
                        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm,
                                            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.Linear)):
                            if hasattr(m, 'weight'):
                                if m.weight.dim() == 1:
                                    nn.init.constant_(m.weight, 1.0)
                                else:
                                    cls(m.weight)
                        if hasattr(m, 'bias'):
                            nn.init.constant_(m.bias, 0.0)

                    model.apply(weight_initializer)
                    print('\t[ConfigReader::load_all] model initialized')
                elif key == 'loss' and isinstance(cls, partial):
                    loss = cls(device=device)
                    setattr(ts, key, loss)
                elif hasattr(ts, key):
                    setattr(ts, key, cls)

        if isinstance(ts.loss_function, UncertaintyWeightedLoss):
            ts.optimizer.add_param_group({
                'params': [ts.loss_function.log_var1, ts.loss_function.log_var2],
                'lr': ts.optimizer.param_groups[0]['lr']
            })

        # Instantiate Dataloaders
        dl_train, dl_val, dl_test = None, None, None
        if 'dataloader' in unused_keys:
            resolved_dls = cr.resolve_singleton(unused_keys['dataloader'], return_top=True,
                                                deferred_args=dict(device=device))
            resolved_dls = resolved_dls if type(resolved_dls) == list else [resolved_dls]
            # noinspection PyTypeChecker
            dl_train, dl_test, dl_val = resolved_dls + [None] * (3 - len(resolved_dls))

        # Return everything
        return model, ts, dl_train, dl_val, dl_test, cr

    @staticmethod
    def load_all_from_checkpoint_grid_search(grid: Dict[str, list], load_kwargs):
        def _to_iterable(obj: Any):
            return obj if type(obj) in [list, tuple] else [obj]

        gvs = [_to_iterable(v) for v in grid.values()]
        for grid_values in itertools.product(*gvs):
            hparams_dict = dict(zip(grid.keys(), grid_values))
            load_kwargs['override_keys'] = hparams_dict
            hp_indices = tuple([hpv.index(v) for hpv, v in zip(gvs, hparams_dict.values())])
            load_kwargs['chkpt_suffix'] = f'_{"".join([str(_) for _ in hp_indices])}'
            yield ConfigReader.load_all_from_checkpoint(**load_kwargs), hparams_dict, hp_indices

    @staticmethod
    def load_model_from_checkpoint(model_key: str, model_config: str, device: str = 'cpu', chkpt_suffix: str = ''):
        # 1) Open checkpoint and load config
        chkpt_fname = f'{model_key}_{model_config}{chkpt_suffix}.pth'.lower()
        chkpt_fpath = os.path.join(checkpoints_path, chkpt_fname.replace(str(checkpoints_path), ''))
        if not os.path.isfile(chkpt_fpath) or not os.path.exists(chkpt_fpath):
            raise FileNotFoundError(chkpt_fpath)
            # return ConfigReader.load_all(model_key=model_key, model_config=model_config, train_config=train_config,
            #                              device=device, **load_kwargs)
        chkpt = torch.load(chkpt_fpath)
        assert 'config' in chkpt.keys(), \
            '\t[ConfigReader::load_all_from_checkpoint] checkpoint error: No "config" key found'
        config_dict = chkpt.pop('config')
        cr = ConfigReader.from_sd(config_dict)
        cr.chkpt_fpath = chkpt_fpath

        # 2) Initialize model/optim/sched/dls from config
        # Instantiate Model
        model, cr = ConfigReader.load_model(cr=cr)
        model.load_state_dict(chkpt.pop('model'))
        model.trained = True
        return model.to(device)

    @staticmethod
    def load_all_from_checkpoint(model_key: str, model_config: str, device: str = 'cpu', chkpt_suffix: str = '',
                                 train_config: Optional[str] = None, gs_dict: Optional[Dict[str, list]] = None,
                                 **load_kwargs) -> CLAR or Generator[CLAR, None, None]:
        if gs_dict is not None:
            extra_kwargs = {'model_key': model_key, 'model_config': model_config, 'device': device,
                            'chkpt_suffix': chkpt_suffix, 'train_config': train_config}
            return ConfigReader.load_all_from_checkpoint_grid_search(gs_dict, {**load_kwargs, **extra_kwargs})

        # 1) Open checkpoint and load config
        chkpt_fname = f'{model_key}_{model_config}{chkpt_suffix}.pth'.lower()
        chkpt_fpath = os.path.join(checkpoints_path, chkpt_fname.replace(str(checkpoints_path), ''))
        if not os.path.isfile(chkpt_fpath) or not os.path.exists(chkpt_fpath):
            raise FileNotFoundError(chkpt_fpath)
            # return ConfigReader.load_all(model_key=model_key, model_config=model_config, train_config=train_config,
            #                              device=device, **load_kwargs)
        chkpt = torch.load(chkpt_fpath)
        assert 'config' in chkpt.keys(), \
            '\t[ConfigReader::load_all_from_checkpoint] checkpoint error: No "config" key found'
        config_dict = chkpt.pop('config')
        cr = ConfigReader.from_sd(config_dict)
        cr.chkpt_fpath = chkpt_fpath

        # 2) Initialize model/optim/sched/dls from config
        load_kwargs.setdefault('device', device)
        load_kwargs.setdefault('train_config', train_config)
        model, ts, dl_train, dl_val, dl_test, cr = ConfigReader.load_all(cr=cr, **load_kwargs)

        # 3) Load states
        #   3.1) model
        model.load_state_dict(chkpt.pop('model'))
        model.trained = True
        loaded_keys = ['model']
        #   3.2) optimizer
        try:
            loaded_keys.extend(
                ts.load_state(chkpt)
            )
        except ValueError:
            ts.optimizer.add_param_group({
                'params': [torch.tensor(0.), torch.tensor(0.)],
                'lr': ts.optimizer.param_groups[0]['lr']
            })
            loaded_keys.extend(
                ts.load_state(chkpt)
            )
        #   3.3) dataloaders
        #        Nothing, since we are using stateless dataloaders.
        #   3.4) loss
        # if 'loss' in chkpt.keys():
        #    ts.loss_function.load_state_dict(chkpt.pop('loss'))
        #    loaded_keys.append('loss')
        print(f'\t[ConfigReader::load_all_from_checkpoint] Loaded state dicts from: {os.path.basename(chkpt_fpath)}')
        print(f'\t[ConfigReader::load_all_from_checkpoint] State dict keys used: {loaded_keys}')

        # Final tasks
        model = model.to(device)
        return model, ts, dl_train, dl_val, dl_test, cr


# ==================================


# noinspection PyUnusedLocal
def capture_checkpoint(model: nn.Module, ts: TrainingSetup, cr: ConfigReader, chkpt_suffix,
                       neptune_run: NeptuneRun) -> dict:
    # TODO alter this to account for separate pretrained/fine-tuned chkpts
    os.makedirs(checkpoints_path, exist_ok=True)
    # 1) Capture state dicts
    chkpt = ts.state_dict()
    chkpt['model'] = model.state_dict()
    chkpt['loss'] = ts.loss_function.state_dict()
    # 2) Retrieve config
    chkpt['config'] = cr.state_dict()
    # 3) Create final file
    filename = os.path.basename(cr.chkpt_fpath)
    filename_with_path = os.path.join(str(checkpoints_path), filename.replace('.pth', chkpt_suffix + '.pth'))
    torch.save(chkpt, filename_with_path)
    print(f'[capture_checkpoint] {filename_with_path}')
    neptune_run.upload_checkpoint(filename_with_path)
    return chkpt


if __name__ == '__main__':
    vit = ConfigReader.load_all(
        model_key='ViT',
        model_config='12-16',
        train_config='default'
    )[0]
    print(vit)
