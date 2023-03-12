import sys
import os

import torch
from torch import nn

from model.mae import ClassifierEncoder
from model.probe import Probe
from utilities.config import ConfigReader
from tqdm.auto import tqdm


def load_checkpoint(filename: str, device: str):
    if not os.path.isfile(filename) or not os.path.exists(filename):
        raise FileNotFoundError(filename)

    chkpt = torch.load(filename)
    assert 'config' in chkpt.keys(), \
        '\t[ConfigReader::load_all_from_checkpoint] checkpoint error: No "config" key found'
    config_dict = chkpt.pop('config')
    cr = ConfigReader.from_sd(config_dict)

    # load_kwargs.setdefault('train_config', "default")
    model, ts, dl_train, dl_val, dl_test, cr = ConfigReader.load_all(cr=cr, device=device, seed=42)

    pretrained = ClassifierEncoder(model, freeze_n_layers=cr.train_config['n_freeze'])

    probe = Probe(
        d_model=model.d_model,
        normalize=True,
        num_classes=dl_train.n_classes,
        device=ts.device)

    net = nn.Sequential(pretrained, probe).to(ts.device)

    # 3) Load states
    #   3.1) model
    net.load_state_dict(chkpt.pop('model'))

    # Final tasks
    net = net.to(device)
    return net, ts, dl_train, dl_val, dl_test, cr

def main(filename):
    device =  'cuda' if torch.cuda.is_available() else 'cpu'

    net, training_setup, dl_train, dl_val, dl_test, cr = load_checkpoint(filename, device)

    dl_val = dl_val if dl_val is not None else dl_test

    print(f'\t[main] Using device: "{training_setup.device}"')

    val_pbar = tqdm(dl_val, desc="Evaluating on validation set", leave=False)
    correct = 0

    # Validation loop
    for i, (x, y) in enumerate(val_pbar):
        if type(x) == list:
            x = x[0]
        x = x.to(training_setup.device)
        y = y.to(training_setup.device)
        with torch.no_grad():
            pred = net(x)
            # noinspection PyUnresolvedReferences
            correct += (pred.argmax(dim=-1) == y.argmax(dim=-1)).sum()

    accuracy = correct / len(dl_val.dataset)

    print(f"validation_accuracy: {accuracy:.3f}|")


if __name__ == "__main__":
    main(sys.argv[1])
