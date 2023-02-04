import argparse

import torch
from torch import nn

import train
import dinosaur
from model.mae import ClassifierEncoder
from model.probe import Probe
from utilities.config import ConfigReader, capture_checkpoint
from utilities.neptune import NeptuneRun
from utilities.train import TrainingSetup, IPretrainer
from utilities.visualize import visualize_reconstruction


def main(args):
    # Load MAE/Optim/lossFunction/Scheduler/Dataloaders/Seed from config
    loading_method = ConfigReader.load_all_from_checkpoint if args.resume else ConfigReader.load_all
    if args.train_config.endswith('gridsearch'):
        HPARAMS_GRID = {
            'MAE.masking_ratio': [0.65, 0.75, 0.80],
        }
        grid_iter = loading_method(gs_dict=HPARAMS_GRID, **vars(args))
        for (mae_model, training_setup, dl_train, dl_val, dl_test, cr), hp, hp_indices in grid_iter:
            print(f'\t[main] Using device: "{training_setup.device}"')
            train_and_save_checkpoint(mae_model, training_setup, dl_train,
                                      dl_val if dl_val is not None else dl_test, cr,
                                      intention=args.intention,
                                      cli_args=args,
                                      chkpt_suffix=f'_{"".join([str(_) for _ in hp_indices])}')
    elif args.intention == "dino":
        model, training_setup, dl_train, dl_val, dl_test, cr = loading_method(**vars(args))
        print(f'\t[main] Using device: "{training_setup.device}"')
        # Change seed as to create two different models
        args.seed = 1000
        model2, _, _, _, _, _ = loading_method(**vars(args))

        dinosaur.run(model, model2, training_setup.device, cr, training_setup)
        return
    else:
        model, training_setup, dl_train, dl_val, dl_test, cr = loading_method(**vars(args))
        print(f'\t[main] Using device: "{training_setup.device}"')

        # Train
        if not (model.trained and args.intention == 'pretrain'):
            train_and_save_checkpoint(model, training_setup, dl_train, dl_val if dl_val is not None else dl_test,
                                      cr, intention=args.intention, cli_args=args)

        # Evaluate on test set
        # TODO

        # Final Visualization
        if args.intention == 'pretrain':
            visualize_reconstruction(model, dl_train, vis_transforms=dl_train.vis_transforms, show=True,
                                     device=training_setup.device)


def train_and_save_checkpoint(model: nn.Module or IPretrainer, training_setup: TrainingSetup, dl_train, dl_val,
                              cr: ConfigReader, intention: str = 'pretrain', chkpt_suffix: str = '',
                              cli_args=None) -> None:
    # Setup logging
    with NeptuneRun() as neptune_run:
        for key, value in (
                ("config/model", cr.yaml_dict),
                ("config/grid_params", cr.override_keys),
                ("config/training", cr.train_config),
                ("config/cli_args", cli_args.__dict__ if cli_args is not None else {})):
            neptune_run.store(key, value)

        if intention == 'pretrain':
            # TODO: Refactor to take checkpoints during training at proper place
            def capture(model, training_setup, epoch_num):
                capture_checkpoint(model, training_setup, cr, f"_{intention}{chkpt_suffix}_{epoch_num}", neptune_run)

            # Pre-Train (MAE)
            train.pre_train(model, training_setup, dl_train, dl_val, neptune_run, capture)
            # Save model checkpoint
            capture_checkpoint(model, training_setup, cr, chkpt_suffix, neptune_run)

        elif intention == "fromscratch" and cli_args.model_key == "squeeze":
            train.fine_tune(model, training_setup, dl_train, dl_val, neptune_run, None)
            capture_checkpoint(model, training_setup, cr, f"_{intention}{chkpt_suffix}", neptune_run)

        elif intention == 'finetune' or intention == 'fromscratch':
            pretrained = ClassifierEncoder(model, freeze_n_layers=cr.train_config['n_freeze'])

            probe = Probe(
                d_model=model.d_model,
                normalize=True,
                num_classes=dl_train.n_classes,
                device=training_setup.device)

            net = nn.Sequential(pretrained, probe).to(training_setup.device)

            training_setup.reset_optimizer(new_model=net)

            # TODO: Refactor to take checkpoints during training at proper place
            def capture(model, training_setup, epoch_num):
                capture_checkpoint(model, training_setup, cr, f"_{intention}{chkpt_suffix}_{epoch_num}", neptune_run)

            train.fine_tune(net, training_setup, dl_train, dl_val, neptune_run, capture)
            capture_checkpoint(net, training_setup, cr, f"_{intention}{chkpt_suffix}", neptune_run)

        elif intention == 'linearprobing':
            pretrained = ClassifierEncoder(model, freeze_n_layers='all')
            probe = Probe(d_model=model.d_model,
                          normalize=True,
                          num_classes=dl_train.n_classes,
                          device=training_setup.device)
            net = nn.Sequential(pretrained, probe).to(training_setup.device)
            training_setup.reset_optimizer(new_model=net)
            train.fine_tune(net, training_setup, dl_train, dl_val, neptune_run)
            capture_checkpoint(net, training_setup, cr, f"_{intention}{chkpt_suffix}", neptune_run)

        else:
            raise ValueError("unrecognized intention")


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description='Entry point for Training/Evaluating PMAE')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_key', type=str, default='MAE')
    parser.add_argument('--model_config', type=str, default='e7d2_128')
    parser.add_argument('--train_config', type=str, default='default')
    parser.add_argument('--intention', type=str, default='pretrain')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    _args = parser.parse_args()
    _args.intention = _args.intention.replace('-', '')
    _args.train_config = f'{_args.intention}_{_args.train_config}'
    # Call main
    main(_args)
