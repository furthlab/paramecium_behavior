# Paramecium skinnerbox

## Installation

Create environment:
```sh
conda create --name napari-env
conda activate napari-env
conda install -c conda-forge napari   
conda install opencv
```

Install Tensorflow for macOS M1/M2:
```sh
pip install tensorflow-macos
pip install tensorflow-metal
```

Install stardist for cell segmentation:
```sh
pip install gputools
pip install stardist
pip install csbdeep
```

Install tapir/tapnet (uses chex version 0.18 to be compatible with stardist stack).
```
git clone https://github.com/deepmind/tapnet.git
cd tapnet
pip install -r ../requirements_tapir.txt
mkdir tapnet/checkpoints
curl -o checkpoints/causal_tapir_checkpoint.npy https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy
cd ..
ls tapnet/checkpoints
```

## Training

### Augment training data set

```sh
python augment_image_data.py
```
This expands `images_org` and `masks_org` into `images` (input) and `masks` (ground truth). 
Input and ground truth are matched based on file name. Format is 8-bit monochrome TIF on both. 

If more than 255 cells needs to be segmented within a single image you can simply change mask format to 16-bit.

### Perform training

```sh
python training.py
```
Open up  tensorboard to follow the results:
```sh
tensorboard --logdir=.
```

## Prediction


