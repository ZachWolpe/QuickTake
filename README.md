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

### Expected Use

`Gender` and `age` determination models are trained on faces. They work fine on a larger image, however, will fail to make multiple predictions in the case of multiple faces in a single image.

The API is currently designed to chain models:

1. `yolo` is used to identify objects.
2. `IF` a person is detected, the `gender` and `age` models are used to make predictions.

This is neatly bundled in the `QuickTake.yolo_loop()` method.

#### Getting Started

Launch a webcam stream:

```python
QL = QuickTake()
QL.launchStream()
```

_*Note*_: Each model returns the results `results_` as well as the runtime `time_`.

Run on a single frame:

```python
from IPython.display import display
from PIL import Image
import cv2

# example images
img = './data/random/dave.png'

# to avoid distractions
import warnings
warnings.filterwarnings('ignore')

# init module
from quicktake import QuickTake
qt = QuickTake()

# extract frame from raw image path
frame = qt.read_image(img)
```

We can now fit `qt.age(<frame>)` or `qt.gender(<frame>)` on the frame. Alternatively we can cycle through the objects detected  by `yolo` and if a person is detected, fit `qt.age()` and `qt.gender()`:

```python
# generate points
for _label, x0,y0,x1,y1, colour, thickness, results, res_df, age_, gender_ in qt.yolo_loop(frame):
    _label = QuickTake.generate_yolo_label(_label)
    QuickTake.add_block_to_image(frame, _label, x0,y0,x1,y1, colour=colour, thickness=thickness)
```

The result is an image with the bounding boxes and labels, confidence (in yolo prediction), age, and gender if a person is detected.

![Example output: a person is detected and thus age, gender are estimated](https://github.com/ZachWolpe/QuickTake/blob/main/data/output_frames/result_dav_2.png).

The staged output is also useful:

![Example of the `YoloV5` detection boundaries](https://github.com/ZachWolpe/QuickTake/blob/main/data/output_frames/result_ct_2.png).


For a more comprehensive _example_ directory. 


## Future

I have many more models; deployment methods & applications in the pipeline.

If you wish to contribute, please email me _@zachcolinwolpe@gmail.com_.