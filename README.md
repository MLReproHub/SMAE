# Semantic Masked Autoencoders

## Table of Contents
1. [Development Setup](#setup)
2. [How To Run](#run)
3. [Model Checkpoints](#checkpoints)

## Development Setup <a href="#setup"></a>
### Setting up environment
#### pip
To set up using pip, run the following:

```bash
cd <project_root>
pip install -r requirements.txt
```

#### conda
To set up using conda, run the following:

```bash
cd ./environment
conda env create -f [os].yaml
```

### Datasets
For our experiments we made use of two datasets:
- **Tiny-ImageNet (TIN)**: gets automatically downloaded from [hugging-face](https://huggingface.co/datasets/Maysee/tiny-imagenet "hugging-face") into `data/tiny-imagenet-200` directory. The latter is also created from our dataloader.
- **CUB-200 (CUB)**: gets automatically downloaded from [caltech.edu](https://www.vision.caltech.edu/datasets/cub_200_2011/) into `data/cub-200-2011` directory. The latter is also created from our dataloader.

### Model Checkpoints <a href="#checkpoints"></a>
Below we provide checkpoints for our trained models on the corresponding datasets. "GD" and "OD" refer to GoogleDrive and OneDrive respectively. Please make sure to place those inside the directory `checkpoints` under project root.

| Model  | Pre-Train  | Fine-Tune  | Link |
| ------------ | ------------ | ------------ | ------------ |
| SqueezeNet  | - | TIN | [GD](http://drive.google.com "GD") [OD](http://onedrive.microsoft.com "OD") |
| MAE/ViT-Lite | TIN  | TIN | [GD](http://drive.google.com "GD") [OD](http://onedrive.microsoft.com "OD")  |
| MAE/ViT-Lite | TIN  | CUB | [GD](http://drive.google.com "GD") [OD](http://onedrive.microsoft.com "OD")  |
| SMAE/ViT-Lite | TIN  | TIN | [GD](http://drive.google.com "GD") [OD](http://onedrive.microsoft.com "OD")  |

Feel free to contact any of the authors if the links below stop working.

### Project Structure
Under project root one can find the following directory tree:


📦`checkpoints`: model checkpoints

📂`config`: config files

&nbsp;&nbsp;┣  📂`model`: for different model versions

&nbsp;&nbsp;┗  📂`train`: for different training hyperparameters

📂`data`

&nbsp;&nbsp;┣  📦`cub-200-2011`: CUB dataset files

&nbsp;&nbsp;┗  📦`tiny-imagenet-200`: TIN dataset files

📂`src`: Sources root. *Make sure it is in PYTHONPATH or you start inside.*

 ┣ 📂`dataset`: dataloader classes

 ┣ 📂`loss`: loss classes

 ┣ 📂`model`: model classes and their dependencies

 ┣ 📂`utilities`: utility methods and classes

 ┃📜...

 ┃📜`evaluate.py`: model evaluation entry point
 
 ┗📜`main.py`: training entry point
 
 
## How To Run <a href="#run"></a>
After making sure that the developmental environment is setup and that SqueezeNet checkpoint exists in the `checkpoints` directory, you can train a MAE model using the `main.py` script. This should be run from inside the `src` directory.

To evaluate a trained model from a saved checkpoint, you can use the `evaluate.py` script. This should be run from inside the `src` directory.

### CLI arguments
The `main.py` script accepts the following CLI arguments (enumerating all the possible values):
```
usage: main.py 
	[--model_key MODEL_KEY]         mae, mae-blockmask, squeeze
	[--model_config MODEL_CONFIG]   e7d2_128, e12d2_128, 200 (for SqueezeNet)
	[--train_config TRAIN_CONFIG]   default, cub
	[--intention INTENTION]         pretrain, finetune, linearprobing, fromscratch, dino
	[--resume]                      <add it to the CLI to resume from the last found checkpoint or the pretraining one (in case intention is finetune/linearprobing)>
	[--seed SEED]                   <SRNG seed>
	[--device DEVICE]               cuda
```

Please see `src/main.py` for more detailed description.


The `evaluate.py` script accepts the following CLI arguments
```
usage: evaluate.py CHECKPOINTFILE	filename, including path, to saved checkpoint to evaluate
```


 ### Example Runs
- **Self-Supervised Pre-Training**
	- Pixel Loss, Random Masking:<br>```python main.py --intention pretrain --train_config default --model_key mae```
	- Combined Perceptual + Pixel Loss (uncertainty-based weighting), Block Masking:<br>```python main.py --intention pretrain --train_config default_uncertainty --model_key mae-blockmask```
	- Combined Perceptual + Pixel Loss (fixed weights), Block Masking:<br>`python main.py --intention pretrain --train_config default_combined --model_key mae-blockmask`
 
 
 - **Supervised Fine-Tuning**
 	- On TIN:<br>`python main.py --intention finetune --model_key mae --train_config default --resume`
 
 - **Transfer Learning**
 	- On CUB:<br>`python main.py --intention finetune --model_key mae --train_config cub --resume`

- **Evaluate Trained Model**
	- `python evaluate.py ../checkpoints/mae_e7d2_128_finetune.pth`

 
### Experiments
- To generatate reconstructions of pre-trained models please consult:<br>`src/experiments/create_figure_2.py`
- &lt;we will update this list with new experiments&gt;

 
