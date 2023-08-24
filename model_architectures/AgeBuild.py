'''
========================================================
Age Build

Wrapper to instantiate Age | SSRNet transfer learning model, load weights & run inference.

: zach wolpe, 22 Aug 23
: zach.wolpe@medibio.com.au
========================================================
'''


from model_architectures.SSRNet import *
from modules.dependencies       import *
from modules.torch_engine      import *



class TorchEngineAgePrediction(TorchEngine):
    # note: magic values can be explained here: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
    _normalize_mean = [0.485, 0.456, 0.406]
    _normalize_std  = [0.229, 0.224, 0.225]


    def __init__(self, model_weights="./model_weights/model_Adam_MSELoss_LRDecay_weightDecay0.0001_batch50_lr0.0005_epoch90_64x64.pth", transform=None):
        self.device = self.fetch_device()

        # init model + weights
        self.model_weights  = model_weights
        self.model          = SSRNet().to(self.device)
        self.name           = SSRNet.__name__
        self.load_weights   = torch.load(model_weights, map_location=torch.device(self.device))
        self.model.load_state_dict(self.load_weights['state_dict'])
        self.model.eval()

        print(f'{self.name} initialized at {self}. Running on device = {self.device}')

    def training_transforms(self):
        raise ValueError('Model is pretrained. Training is not implemented.')

    def inference_transforms(self, input_size_=64):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size_, input_size_)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TorchEngineAgePrediction._normalize_mean, TorchEngineAgePrediction._normalize_std)])


    def inference_pixels(self, image_:np.ndarray, input_size_=64):
        # inference on a single image.
        start_time_ = time.time()
        transforms_ = self.inference_transforms(input_size_=input_size_)
        image_      = transforms_(image_)
        image_      = image_[np.newaxis, ]
        image_      = image_.to(self.device)
        with torch.no_grad():
            results_ = self.model(image_)
        return results_,  time.time() - start_time_


    def inference(self, image_path_: str = None, images_dir_: list[str] = None, image_pixels_ : torch.Tensor = None, input_size_=64):

        if images_dir_ is not None:
            start_time_ = time.time()
            preds_      = super().inference(image_path_, images_dir_, image_pixels_, input_size_)
            return preds_, time.time() - start_time_

        elif image_path_ is not None:
            # Single image path
            # NOTE: This will propagrate to all images when run in a directory.
            yhat_, time_ = super().inference(image_path_, images_dir_, image_pixels_, input_size_)
            # yhat_           = self.transform_prediction_to_label(results_)
            return yhat_, time_
    
        elif image_pixels_ is not None:
            # Single image pixels
            yhat_, time_ = self.inference_pixels(image_pixels_, input_size_)
            return yhat_, time_

        raise ValueError('Must provide either an single image (image_path_) or a directory of images (images_dir_)')




