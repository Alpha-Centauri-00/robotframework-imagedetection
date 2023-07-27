from .Base_detection import Base_Detect
from robot.api.deco import keyword,library
from robot.api import logger
import os

ROBOT_LIBRARY_SCOPE = 'SUITE'

class Imagedetection:
    def __init__(self):
        self.Base = Base_Detect()
        self.log_image_width = 150
        self.log_image_height = 100
    
        self.save_path = "captured_photo.jpg"

    def _show_photo_in_log(self,_path):
        Log_photo = f'<img src="{_path}" width="{self.log_image_width}" height="{self.log_image_height}">'
        logger.info(Log_photo,html=True,also_console=False)

    @keyword
    def Train_Model(self,training_path,validation_path):
        self.Base.train_model(training_path,validation_path)

    @keyword
    def detect_from_path(self,model_name,photo_path):
        self.Base._predict(model_name,photo_path)
        self._show_photo_in_log(_path=photo_path)
        
    @keyword
    def detect_from_webcam(self,model_name):
        self.Base.capture_and_predict(model_name,self.save_path)

        curdir = os.getcwd()
        current_directory_with_double_backslashes = curdir.replace("\\", "\\\\")

        self._show_photo_in_log(rf"{current_directory_with_double_backslashes}\\{self.save_path}")
        
    @keyword
    def detect_from_google(self,model,photo):
        self.Base.predict_from_google_model(model,photo)
        self._show_photo_in_log(photo)
