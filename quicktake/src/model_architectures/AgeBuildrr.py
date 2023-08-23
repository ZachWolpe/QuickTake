'''
========================================================
SSRNet Build

Wrapper to instantiate SSRNet model, load weights & run inference.

: zach wolpe, 18 Aug 23
: zach.wolpe@medibio.com.au
========================================================
'''

from model_architectures.SSRNet import *
from modules.dependencies       import *

class TorchEngineAgePrediction:
    # note: magic values can be explained here: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
    _normalize_mean = [0.485, 0.456, 0.406]
    _normalize_std  = [0.229, 0.224, 0.225]


    def __init__(self, model_weights="./model_weights/model_Adam_MSELoss_LRDecay_weightDecay0.0001_batch50_lr0.0005_epoch90_64x64.pth", transform=None):
        self.device = (
            "cuda"      if torch.cuda.is_available() 
            else "mps"  if torch.backends.mps.is_available()
            else "cpu"
            )


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

    def inference(self, image_path_:str=None, images_dir_:list[str]=None, input_size_=64):
        if images_dir_ is not None:
            # iterate over images in directory
            results_list = []
            for image in os.listdir(images_dir_):
                try:
                    age_, _ = self.inference(image_path_=os.path.join(images_dir_, image), input_size_=input_size_)
                    results_list.append(age_.tolist()[0])
                    pd_result = pd.DataFrame(results_list)
                except:
                    print('skipping image: {image}')
            pd_result.columns = ['yhat']
            return pd_result
        
        elif image_path_ is not None:
            # inference on a single image.
            image_      = cv2.imread(image_path_)
            start_time_ = time.time()
            transforms_ = self.inference_transforms(input_size_=input_size_)
            image_      = transforms_(image_)
            image_      = image_[np.newaxis, ]
            image_      = image_.to(self.device)
            with torch.no_grad():
                results_ = self.model(image_)
            return results_,  time.time() - start_time_
        
        raise ValueError('Must provide either an single image (image_path_) or a directory of images (images_dir_)')


    def inference_transforms(self, input_size_=64):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size_, input_size_)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TorchEngineAgePrediction._normalize_mean, TorchEngineAgePrediction._normalize_std)])






# # Example use:

# # instantiate model
# torch_model_age = TorchEngineAgePrediction()

# # inference: single image
# image_path, input_size = './data/random/IMG_0483.jpeg', 64
# age_, cost_time = torch_model_age.inference(image_path_=image_path)
# print(f'Age: {age_}, Cost time: {cost_time}')


# # inference: directory of images
# image_file_path = './data/random/'
# pd_result = torch_model_age.inference(image_path_=None, images_dir_=image_file_path, input_size_=input_size)
# print(pd_result.describe())
# print(pd_result['yhat'].value_counts())