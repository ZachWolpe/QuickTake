'''
========================================================
Torch Helpers

Methods and transformations to use with PyTorch models.

: zach wolpe, 20 Aug 23
: zach.wolpe@medibio.com.au
========================================================
'''

from model_architectures.SSRNet import *
from modules.dependencies       import *

class TorchHelpers:
    # note: magic values can be explained here: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
    _normalize_mean = [0.485, 0.456, 0.406]
    _normalize_std  = [0.229, 0.224, 0.225]

    @staticmethod
    def fetch_device():
        return (
            "cuda"      if torch.cuda.is_available() 
            else "mps"  if torch.backends.mps.is_available()
            else "cpu"
            )

    @staticmethod
    def read_image(image_path):
        return cv2.imread(image_path)

    @staticmethod
    def inference_transforms(input_size_=64):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size_, input_size_)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TorchHelpers._normalize_mean, TorchHelpers._normalize_std)])

    def inference_pixels(self, image_:np.ndarray, input_size_=64):
        # inference on a single image.
        start_time_ = time.time()
        transforms_ = self.inference_transforms(input_size_=input_size_)
        image_      = transforms_(image_)
        image_      = image_[np.newaxis, ]
        image_      = image_.to(self.device)
        print('device', self.device)
        with torch.no_grad():
            results_ = self.model(image_)
        return results_,  time.time() - start_time_
    

    def inference(self, image_path_:str=None, images_dir_:list[str]=None, image_pixels_:torch.Tensor=None, input_size_=64):
        if images_dir_ is not None:
            # iterate over images in directory
            results_list = []
            for image in os.listdir(images_dir_):
                try:
                    pred_, _ = self.inference(image_path_=os.path.join(images_dir_, image), input_size_=input_size_)
                    if isinstance(pred_, torch.Tensor):
                        results_list.append(pred_.tolist()[0])
                    else:
                        results_list.append(pred_)
                    pd_result = pd.DataFrame(results_list)
                except:
                    print('skipping image: {image}')
            if pd_result.shape[1] == 1:
                pd_result.columns = ['yhat']
            else:
                pd_result.columns = ['yhat_' + str(i) for i in range(pd_result.shape[1])]
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
        
        elif image_pixels_ is not None:
            start_time_ = time.time()
            transforms_ = self.inference_transforms(input_size_=input_size_)
            image_      = transforms_(image_pixels_)
            image_      = image_[np.newaxis, ]
            image_      = image_.to(self.device)
            with torch.no_grad():
                results_ = self.model(image_)
            return results_,  time.time() - start_time_
        
        raise ValueError('Must provide either an single image (image_path_) or a directory of images (images_dir_)')

