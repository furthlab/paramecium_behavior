# Prerequisites

Create environment:
```sh
conda create --name napari-env
conda activate napari-env
conda install -c conda-forge napari   
conda install opencv
```

Install Tensorflow and stardist for macOS M1/M2:
```sh
pip install tensorflow-macos
pip install tensorflow-metal

pip install gputools
pip install stardist
pip install csbdeep
```

TAPIR:

Cant use TAPIR yet due to conflicting numpy versions with the rest of the stack.

```
git clone https://github.com/deepmind/tapnet.git
cd tapnet
pip install -r ../requirements_tapir.txt
mkdir tapnet/checkpoints
curl -o checkpoints/causal_tapir_checkpoint.npy https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy
cd ..
ls tapnet/checkpoints
```

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow-macos 2.13.0 requires numpy<=1.24.3,>=1.22, but you have numpy 1.25.1 which is incompatible.
numba 0.57.1 requires numpy<1.25,>=1.21, but you have numpy 1.25.1 which is incompatible.
gputools 0.2.14 requires numpy<1.24.0, but you have numpy 1.25.1 which is incompatible.
```

# Training

## Create training data set

```sh
python augment_image_data.py
```

## Perform training

```sh
python training.py
```



