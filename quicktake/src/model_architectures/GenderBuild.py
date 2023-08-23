'''
========================================================
ResNet 18 Transfer Learning Gender Model Build.

Transfer Learning:  Use ResNet18 pretrained on ImageNet.
Application:        Retrain the prediction layer to predict gender.

# REFERENCE:
    Model adapted from:
    ## Dataset
    [Gender classification dataset](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset).

    ## Reference
    Code taken from [here](https://github.com/ndb796/Face-Gender-Classification-PyTorch/blob/main/Face_Gender_Classification_using_Transfer_Learning_with_ResNet18.ipynb).

Many thanks to the original author!

----
: zach wolpe, 20 Aug 23
: zach.wolpe@medibio.com.au
========================================================
'''


from modules.dependencies  import *
from modules.torch_helpers import *


class TorchEngineGenderPrediction(TorchHelpers):

    def __init__(self, model_weights='./model_weights/face_gender_classification_transfer_learning_with_ResNet18.pth') -> None:
        # fetch device
        self.device             = self.fetch_device()
        self.prediction_classes = ['female', 'male']

        # load model
        self.model_weights  = model_weights
        self.model          = torchvision.models.resnet18(pretrained=True)
        self.name           = 'ResNet18_Transfer_Gender_Prediction'
        self.num_features   = self.model.fc.in_features
        self.model.fc       = nn.Linear(self.num_features, 2) # binary classification (num_of_class == 2)
        self.model.load_state_dict(torch.load(model_weights, map_location=torch.device(self.device)))
        self.model.to(self.device)
        self.model.eval()

    def transform_prediction_to_label(self, results_: torch.Tensor):
        _, preds_ = torch.max(results_, 1)
        return self.prediction_classes[preds_.item()]
    
    def inference_pixels(self, image_: np.ndarray, input_size_=64):
        pred_, time_ = super().inference(image_pixels_=image_, input_size_=input_size_)
        return self.transform_prediction_to_label(pred_), time_
    

    def inference(self, image_path_: str = None, images_dir_: list[str] = None, image_pixels_ : torch.Tensor = None, input_size_=64):

        if images_dir_ is not None:
            start_time_ = time.time()
            preds_      = super().inference(image_path_, images_dir_, image_pixels_, input_size_)
            return preds_, time.time() - start_time_

        elif image_path_ is not None:
            # Single image path
            # NOTE: This will propagrate to all images when run in a directory.
            results_, time_ = super().inference(image_path_, images_dir_, image_pixels_, input_size_)
            yhat_           = self.transform_prediction_to_label(results_)
            return yhat_, time_
    
        elif image_pixels_ is not None:
            # Single image pixels
            yhat_, time_ = self.inference_pixels(image_pixels_, input_size_)
            return yhat_, time_




# Example usage:
# if __name__ == '__main__':
#     tgp = TorchEngineGenderPrediction()
#     X   = torch.randn(3, 224, 224, requires_grad=False)#.to(tgp.device)
#     image_path  = './data/random/IMG_0431.jpg'
#     image_paths = './data/random/'

#     print('-------------------------------------------------')
#     results_, time_ = tgp.inference_pixels(X)
#     print(results_)
#     print(time_)
#     print('Test (0x1): PASSED!')
#     print('-------------------------------------------------')



#     print('-------------------------------------------------')
#     results_, time_ = tgp.inference(image_path_=image_path)
#     print(results_)
#     print(time_)
#     print('Test (0x2): PASSED!')
#     print('-------------------------------------------------')


#     print('-------------------------------------------------')
#     results_, time_ = tgp.inference(images_dir_=image_paths)
#     print(results_)
#     print(time_)
#     print('Test (0x3): PASSED!')
#     print('-------------------------------------------------')
