# QuickTake

Off-the-shelf computer vision ML models. Yolov5, gender and age determination.

The goal of this repository is to provide, easy to use, abstracted, APIs to powerful computer vision models.


## Models

$3$ models are currently available:

- `Object detection`
- `Gender determination`
- `Age determination`

## Model Engine

The models 
- `YoloV5`: Object detection. This forms the basis of the other models. Pretrained on `COCO`. Documentation [here](https://pjreddie.com/darknet/yolo/)
- `Gender`: `ResNet18` is used as the models backend. Transfer learning is applied to model gender. The additional gender training was done on the [gender classification dataset](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset), using code extract from [here](https://github.com/ndb796/Face-Gender-Classification-PyTorch/blob/main/Face_Gender_Classification_using_Transfer_Learning_with_ResNet18.ipynb).

- `Age`: The age model is an implementation of the `SSR-Net` paper: [SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation](https://www.ijcai.org/proceedings/2018/0150.pdf). The `pyTorch` model was largely derived from [oukohou](https://github.com/oukohou/SSR_Net_Pytorch/blob/master/inference_images.py).

## Getting Started

Install the package with pip:

````pip install quicktake````


## Usage

Build an instance of the class:

```python
from quicktake import QuickTake
```

#### Image Input

Each model is designed to handle $3$ types of input:

- `raw pixels (torch.Tensor)`: raw pixels of a single image. Used when streaming video input.
- `image path (str)`: path to an image. Used when processing a single image.
- `image directory (str)`: path to a directory of images. Used when processing a directory of images.

#### Examples

Launch a webcam stream:

```python
QL = QuickTake()
QL.launchStream()
```

_*Note*_: Each model returns the results `results_` as well as the runtime `time_`.

Example parameters:

```python
image_size  = 64
X           = torch.randn(3, 224, 224, requires_grad=False)#.to(tgp.device)
X_yolo      = torch.randn(1, 3, 224, 224, requires_grad=False)#.to(tgp.device)
image_path  = './data/random/IMG_0431.jpg'
image_paths = './data/random/'
```

Process a torch.Tensor:

```python
results_, time_ = QL.gender(X, new_init=True)
results_, time_ = QL.yolov5(X_yolo, new_init=True)
results_, time_ = QL.age(X, new_init=True)
```

Process a single image (path):

```python
results_, time_ = QL.gender(image_path, new_init=True)
results_, time_ = QL.yolov5(image_path, new_init=True)
results_, time_ = QL.age(image_path, new_init=True)
```

Process a directory of images:

```python
results_, time_ = QL.gender(image_paths, new_init=True)
results_, time_ = QL.yolov5(image_paths, new_init=True)
results_, time_ = QL.age(image_paths, new_init=True)
```



## Future

We have many more models & deployments in the pipeline. If you wish to contribute, please email me @zachcolinwolpe@gmail.com!