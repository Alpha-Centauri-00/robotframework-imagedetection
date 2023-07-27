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
        '''
        - Collect Dataset / Photos:
        Before using the Keyword, make sure you have gathered the dataset or photos you want to work with. This dataset will be the foundation for training and validation.
        
        - Create a "Data" Directory:
        Once you have the dataset, create a directory called "Data" in your project. This directory will serve as the main container for your data.
        
        - Organize Training and Validation Folders:
        Inside the "Data" directory, create two sub-folders: "training" and "validation." These folders will be used to store the data separately for training and validation purposes.
        
        - Classification Folders:
        Inside each of the "training" and "validation" folders, create additional sub-folders representing different classifications or categories for your data. For instance, each classification folder could represent a different object or class.
        
        - Distribute Photos:
        When organizing your dataset, ensure that you distribute the photos among the classification folders inside the "training" and "validation" directories without any duplication between the two sets. Each classification folder should exist separately in both directories, but they should contain different photos to maintain the distinction between training and validation data.

        '''
        self.Base.train_model(training_path,validation_path)

    @keyword
    def detect_from_path(self,model_name,photo_path):
        '''
        Pretrained Model: 
        Before using this Keyword, make sure you have already trained a model. The model name should be provided as the first argument when using this Keyword.

        Model Location: 
        After running the "Train Model" Keyword, the trained model will be saved in your project under the name "model.keras".

        Argument Details: 
        When using the Keyword, provide the model name as the first argument, which should be the path where the model is already saved in your project (e.g., "/path/to/your/model.keras").

        Test Photo Path: 
        The second argument should be the path to the test photo you want to use with the model.

        Photo Class Detection: 
        When you execute the Keyword with the test photo, it will utilize the trained model to detect the class or category to which the photo belongs. The Keyword will provide the result in log.html file.
        '''
        self.Base._predict(model_name,photo_path)
        self._show_photo_in_log(_path=photo_path)
        
    @keyword
    def detect_from_webcam(self,model_name):
        '''
        Pretrained Model: 
        Before using this Keyword, ensure you have already trained a model. The model name should be provided as the model_name argument when calling this Keyword.

        Model Location: 
        After running the "Train Model" Keyword it saves the model as "model.keras", the trained model will be available in your project.

        Webcam Photo: 
        When using "Detect from Webcam" Keyword, it will automatically activate your webcam and take a photo. The taken photo will be automatically saved in your project directory.

        Class Prediction: 
        The function will utilize the trained model to predict the class or category to which the taken photo belongs. It will provide the result, indicating the detected class for the photo taken by the webcam.
        
        '''
        self.Base.capture_and_predict(model_name,self.save_path)

        curdir = os.getcwd()
        current_directory_with_double_backslashes = curdir.replace("\\", "\\\\")

        self._show_photo_in_log(rf"{current_directory_with_double_backslashes}\\{self.save_path}")
        
    @keyword
    def detect_from_google(self,model,photo):
        self.Base.predict_from_google_model(model,photo)
        self._show_photo_in_log(photo)
