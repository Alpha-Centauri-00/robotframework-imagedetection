from .Base_detection import Base_Detect
from robot.api.deco import keyword,library
from robot.api import logger

ROBOT_LIBRARY_SCOPE = 'SUITE'

class Imagedetection:
    def __init__(self):
        self.Base = Base_Detect()

    @keyword
    def Train_Model(self,training_path,validation_path):
        self.Base.train_model(training_path,validation_path)
