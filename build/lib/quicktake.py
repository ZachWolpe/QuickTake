'''
========================================================
QuickTake!

This script houses the package build.

: version: 0.0.1
: zach wolpe, 22 Aug 23
: zach.wolpe@medibio.com.au
========================================================
'''


from model_architectures.AgeBuild       import *
from model_architectures.SSRNet         import *
from model_architectures.GenderBuild    import *
from modules.dependencies               import *
from modules.cv_helpers                 import *
from modules.torch_engine               import *


class QuickTake(TorchEngine, CVHelpers):
    # inherit models
    available_models_       = ['gender', 'age', 'yolov5']
    available_input_types   = [torch.Tensor, np.ndarray, str, list]

    def __init__(self, verbose:bool=False) -> None:
        # instantiate model subclasses - avoid inheretance due to ambiguity that may arise from decouplied methods from their core concrete classes.
        self.model_instances    = {}
        self.verbose            = verbose


    def attach_relative_path(self):
        """Attach relative path to locate model dirs."""
        relative_path     = os.path.abspath(__file__)
        relative_path_2   = '/'.join(relative_path.split('/')[:-1])
        self._gender_path = f"{relative_path_2}/model_weights/face_gender_classification_transfer_learning_with_ResNet18.pth"
        self._age_path    = f"{relative_path_2}/model_weights/model_Adam_MSELoss_LRDecay_weightDecay0.0001_batch50_lr0.0005_epoch90_64x64.pth"

    @staticmethod
    def _inference_check(image:torch.Tensor | str | list[str]=None, model:str=None) -> dict:
        """
        Test the datatype passed to inference.
        """
        if image is not None:
            assert type(image) in QuickTake.available_input_types, f'Image type not supported! Must be torch.Tensor, str, or list[str]. Got {type(image)}'        
        
        if model is not None:
            assert model.lower() in QuickTake.available_models_, f'Unsupported model specification \"{model}\"! Must be one of: {QuickTake.available_models_}'
        
        return True

    def _inference(self, image, model='gender', input_size_:int=64):
        if isinstance(image, torch.Tensor) or isinstance(image, np.ndarray): 
                                            results_, time_ = self.model_instances[model].inference(image_pixels_=image, input_size_=input_size_)
        elif os.path.isdir(image):          results_, time_ = self.model_instances[model].inference(images_dir_=image,   input_size_=input_size_)
        else:                               results_, time_ = self.model_instances[model].inference(image_path_=image,   input_size_=input_size_)
        return results_, time_

    def _instantiate_model(self, model='gender', model_weight_path=None, instance=None, new_init=False):
        if new_init or model not in self.model_instances.keys():
            self.model_instances[model] = instance(model_weights=model_weight_path)

    def yolov5(self, image:torch.Tensor | str, new_init=False):
        if isinstance(image, str) and os.path.isdir(image):
            results     = []
            start_time_  = time.time()
            for img in os.listdir(image):
                # if directory, recursively call yolov5 on each image.
                path = os.path.join(image, img)
                results.append(self.yolov5(path, new_init=new_init))
            return results, time.time() - start_time_
        _model      = 'yolov5'
        start_time_ = time.time()
        _engine     = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, verbose=self.verbose)
        self._inference_check(image)
        self.model_instances[_model] = _engine
        results = self.model_instances[_model](image)
        return results, time.time() - start_time_


    def gender(self, image:torch.Tensor | str | list[str]=None, new_init=False, input_size_:int=64):
        _model   = 'gender'
        _engine  = TorchEngineGenderPrediction
        _weights = self._gender_path
        if self.verbose: print(f'Fitting {_model} model.')
        QuickTake._inference_check(image, _model)
        self._instantiate_model(_model, _weights, _engine, new_init)
        results_, time_ = self._inference(image, _model, input_size_=input_size_)
        if self.verbose: print('Model fit successfully in {} seconds!'.format(time_))
        return results_, time_


    def age(self, image:torch.Tensor | str | list[str]=None, new_init=False, input_size_:int=64):
        _model   = 'age'
        _engine  = TorchEngineAgePrediction
        _weights = self._gender_path

        if self.verbose: print(f'Fitting {_model} model.')
        QuickTake._inference_check(image, _model)
        self._instantiate_model(_model, _weights, _engine, new_init)
        results_, time_ = self._inference(image, _model, input_size_=input_size_)
        if self.verbose: print('Model fit successfully in {} seconds!'.format(time_))
        return results_, time_


    def yolo_loop(self, frame:torch.Tensor, face_threshold=0., _NEW_INIT=False):
        results, _  = self.yolov5(image=frame, new_init=_NEW_INIT)
        res_df      = results.pandas().xyxy[0]
        gyp         = QuickTake.generate_yolo_points(res_df)
        # avoid caching error
        age_        = None
        gender_     = None
        for name, confidence, x0,y0,x1,y1, colour, thickness in gyp:
            _label = [('person', name, False, False), ('confidence', confidence, True, True)]
            
            # extract face
            if name == 'person' and confidence > face_threshold:
                face = frame[y0:y1, x0:x1]

                # inference : age
                age_, _ = self.age(image=face, new_init=_NEW_INIT)
                _label.append(('age', age_.tolist()[0], True, True))

                # inference on gender
                gender_, _ = self.gender(image=face, new_init=_NEW_INIT)
                _label.append(('gender', gender_, False, False))

            yield _label, x0,y0,x1,y1, colour, thickness, results, res_df, age_, gender_
    


    def launchStream(self):
        print('Launching QuickTake!')
        _NEW_INIT = False
        cam = cv2.VideoCapture(0)
        while True:
            check, frame = cam.read()
        
            # Inference ------------------------------------->>
            for _label, x0,y0,x1,y1, colour, thickness, results, res_df, age_, gender_ in self.yolo_loop(frame):       
                # add block and text to image
                _label = QuickTake.generate_yolo_label(_label)
                QuickTake.add_block_to_image(frame, _label, x0,y0,x1,y1, colour=colour, thickness=thickness)
            # Inference ------------------------------------->>

            # stream ---------------------------------------->>
            cv2.imshow('video', frame)
            # stream ---------------------------------------->>

            # breakpoints ----------------------------------->>
            key         = cv2.waitKey(1)
            breaker_    = (key == 27) or (key == ord('q'))
            if breaker_: break
            # breakpoints ----------------------------------->>

        # terminate session
        cam.release()
        cv2.destroyAllWindows()



# example usage
# if __name__ == '__main__':
#     QL = QuickTake()
#     QL.launchStream()

# if False:
#     # test inputs
#     image_size  = 64
#     X           = torch.randn(3, 224, 224, requires_grad=False)#.to(tgp.device)
#     X_yolo      = torch.randn(1, 3, 224, 224, requires_grad=False)#.to(tgp.device)
#     image_path  = './data/random/IMG_0431.jpg'
#     image_paths = './data/random/'

#     print('............................................')
#     print('using torch.Tensor')
#     res, ti = QL.gender(X, new_init=True)
#     print(res); print(ti)
#     res, ti = QL.yolov5(X_yolo, new_init=True)
#     print(res); print(ti)
#     res, ti = QL.age(X, new_init=True)
#     print(res); print(ti)
#     print('............................................')

#     print('............................................')
#     print('using image path')
#     res, ti = QL.gender(image_path, new_init=True)
#     print(res); print(ti)
#     res, ti = QL.yolov5(image_path, new_init=True)
#     print(res); print(ti)
#     res, ti = QL.age(image_path, new_init=True)
#     print(res); print(ti)
#     print('............................................')

#     print('............................................')
#     print('using image directory')
#     res, ti = QL.gender(image_paths, new_init=True)
#     print(res); print(ti)
#     res, ti = QL.yolov5(image_paths, new_init=True)
#     print(res); print(ti)
#     res, ti = QL.age(image_paths, new_init=True)
#     print(res); print(ti)
#     print('............................................')
