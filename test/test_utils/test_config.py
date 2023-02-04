import os.path
import shutil
from collections import OrderedDict
from unittest import TestCase

from torch import nn

from model.mae import MAE
from src.utilities.config import ConfigArgResolver, ConfigReader, RecursiveModuleClassResolver, capture_checkpoint
from utilities.path import checkpoints_path
from utilities.train import TrainingSetup


class TestModule1(nn.Module):
    def __init__(self, arg1, arg2):
        super(TestModule1, self).__init__()
        print(f'arg1={arg1}, arg2={arg2}')
        self.arg1 = arg1
        self.arg2 = arg2


class TestModule2(nn.Module):
    def __init__(self, tm1: TestModule1, tm2_arg1):
        super(TestModule2, self).__init__()
        print(f'tm2_arg1={tm2_arg1}')
        print(f'arg1={tm1.arg1}, arg2={tm1.arg2}')


class TestConfig(TestCase):
    def setUp(self) -> None:
        self.ar = ConfigArgResolver(mapping={'TestModule2.tm1': 'TestModule1'})
        self.config = ConfigReader(cls_resolvers=RecursiveModuleClassResolver(os.path.dirname(__file__)),
                                   arg_resolver=self.ar)
        self.chkpt_path = os.path.join(checkpoints_path, 'test_temp')
        os.makedirs(self.chkpt_path, exist_ok=True)

    def test_resolve(self):
        self.test_load()
        instances = self.config.resolve()
        self.assertEqual(type(instances), OrderedDict)
        self.assertEqual(2, len(instances))
        for instance_cls, instance in instances.items():
            self.assertIn(type(instance), [TestModule1, TestModule2])

    def test_resolve_single(self):
        self.test_load()
        instances = self.config.resolve(only_keys='TestModule2')
        self.assertEqual(type(instances), OrderedDict)
        self.assertEqual(1, len(instances))
        for instance_cls, instance in instances.items():
            self.assertIn(type(instance), [TestModule2])

    def test_load(self):
        yaml_contents = 'TestModule1:\n  arg1: test1\n  arg2: 5.6\nTestModule2:\n  tm2_arg1: test_tm2_arg1\n'
        self.config.load(yaml_contents)
        self.assertEqual(2, len(self.config.entries))
        self.assertEqual(2, self.config.n_unresolved)
        for entry in self.config.entries:
            self.assertIn(entry.class_name, ['TestModule1', 'TestModule2'])

    def test_load_all_and__capture_checkpoint(self):
        # Load checkpoint
        mae, training_setup, dl_train, dl_test, cr = ConfigReader.load_all(
            device='cpu',
            model_key='mae',
            model_config='debug',
            train_config='default',
            args_mapping={'MAE.encoder': 'TransformerEncoder', 'MAE.decoder': 'TransformerDecoder'},
        )
        self.assertEqual(type(mae), MAE)
        self.assertEqual(type(training_setup), TrainingSetup)

        self.assertEqual(2, len(ConfigReader.INSTANCES))

        # Create and store checkpoint
        chkpt_fpath = os.path.join(self.chkpt_path, 'tmp.pth')
        chkpt = capture_checkpoint(model=mae, ts=training_setup, cr=cr)
        self.assertEqual(type(chkpt), dict)
        self.assertIn('config', chkpt.keys())
        self.assertIn('config', chkpt['config'].keys())
        self.assertIn('cls_resolver', chkpt['config'].keys())
        self.assertIn('cls_resolver', chkpt['config'].keys())
        self.assertIn('model', chkpt.keys())
        self.assertIn('optimizer', chkpt.keys())
        self.assertIn('lr_scheduler', chkpt.keys())
        self.assertEqual(type(chkpt['config']), dict)

        cr_chkpt = ConfigReader.from_sd(chkpt['config'])
        cr = ConfigReader.INSTANCES[-1]
        self.assertIsInstance(cr, ConfigReader)
        self.assertIsNotNone(cr.yaml_dict)
        for key in cr.yaml_dict.keys():
            self.assertIn(key, cr_chkpt.yaml_dict.keys())
            if type(cr.yaml_dict[key]) == dict:
                for kk, kv in cr.yaml_dict[key].items():
                    self.assertIn(kk, cr_chkpt.yaml_dict[key].keys())

        # Reload all from checkpoint
        ConfigReader.load_all_from_checkpoint(model_key='mae', model_config='debug')

    def tearDown(self) -> None:
        ConfigReader.INSTANCES = []
        shutil.rmtree(self.chkpt_path)
