from pathlib import Path

from .Base_detection import Base_Detect
from robot.api.deco import keyword,library
from robot.api import logger
import cv2
import torch

ROBOT_LIBRARY_SCOPE = 'SUITE'

class Imagedetection:
    def __init__(self):
        self.Base = Base_Detect()
        self.log_image_width = 150
        self.log_image_height = 150

        self.save_path = "captured_photo.jpg"

    def _show_photo_in_log(self,_path):
        import base64
    
        with open(_path, 'rb') as image:
            image_data = image.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            # Create an HTML img tag with the base64-encoded image data
            img_tag = f'<img src="data:image/png;base64,{image_base64}" alt="Image" width="{self.log_image_width}" height="{self.log_image_height}"/>'
            logger.info(img_tag,html=True,also_console=False)

    @keyword
    def get_device_info(self):
        '''
        Logs and returns which device (CPU or CUDA GPU) will be used for training and detection.

        Useful as a first sanity check to confirm whether an NVIDIA GPU was actually detected
        before running a full training job - if you expected "cuda" but see "cpu", check that
        you installed a CUDA-enabled PyTorch build (see the README's GPU support section) rather
        than the default CPU-only one.

        Returns the device as a string (e.g. "cuda" or "cpu") and writes the full details
        (device, PyTorch version, CUDA version, GPU name if available) to log.html.
        '''
        device = self.Base.device
        details = [
            f"device: {device}",
            f"torch version: {torch.__version__}",
            f"CUDA available: {torch.cuda.is_available()}",
        ]
        if torch.cuda.is_available():
            details.append(f"CUDA version: {torch.version.cuda}")
            details.append(f"GPU name: {torch.cuda.get_device_name(0)}")

        logger.info("<br/>".join(details), html=True)
        return str(device)

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

        Returns the path to the saved model file (e.g. "model_202601011200.pt"), so it can be
        captured into a variable and passed straight to "Detect From Path" without hardcoding
        the generated filename.
        '''
        return self.Base.train_model(training_path,validation_path)

    @keyword
    def detect_from_path(self,model_name,photo_path):
        '''
        Pretrained Model: 
        Before using this Keyword, make sure you have already trained a model. The model name should be provided as the first argument when using this Keyword.

        Model Location:
        After running the "Train Model" Keyword, the trained model will be saved in your project under a name like "model_<timestamp>.pt".

        Argument Details:
        When using the Keyword, provide the model name as the first argument, which should be the path where the model is already saved in your project (e.g., "/path/to/your/model_<timestamp>.pt").

        Test Photo Path: 
        The second argument should be the path to the test photo you want to use with the model.

        Photo Class Detection: 
        When you execute the Keyword with the test photo, it will utilize the trained model to detect the class or category to which the photo belongs. The Keyword will provide the result in log.html file.
        '''
        predict = self.Base._predict(model_name,photo_path)
        self._show_photo_in_log(_path=photo_path)
        return predict
    

    @keyword
    def detect_from_webcam(self,model_name,x=None,y=None,width=None,height=None,camera_index=0):
        '''
        Pretrained Model:
        Before using this Keyword, ensure you have already trained a model. The model name should be provided as the model_name argument when calling this Keyword.

        Model Location:
        After running the "Train Model" Keyword it saves the model as "model_<timestamp>.pt", the trained model will be available in your project.

        Webcam Photo:
        When using "Detect from Webcam" Keyword, it will automatically activate your webcam and take a photo. The taken photo will be automatically saved in your project directory.

        Class Prediction:
        The function will utilize the trained model to predict the class or category to which the taken photo belongs. It will provide the result, indicating the detected class for the photo taken by the webcam.

        Crop Region (x, y, width, height):
        Only the rectangle described by these four arguments is cropped out of the
        live frame and classified - the rest of the frame is ignored. Leave them
        unset (the default) to use whichever ROI was embedded in the model file
        itself when it was trained - annotate.py's --camera mode saves the box
        you drew as last_roi.json, and Train Model embeds it into the checkpoint
        automatically, so the calibrated region travels with the model instead
        of living only in a terminal printout or a hand-copied .robot variable
        that's easy to forget to update after recalibrating. Pass any of the
        four explicitly to override that stored ROI (e.g. while calibrating a
        new one). If the model has no stored ROI at all (trained before this
        existed, or without ever running annotate.py --camera), this falls
        back to an arbitrary (100, 100, 200, 200) box that is almost certainly
        NOT where your subject sits - check log.html's full-frame view (with
        the crop region marked in green) to confirm the box is actually on
        target rather than trusting a confident-looking wrong prediction.

        Camera Index:
        If your machine has more than one camera, pass camera_index to pick which one
        OpenCV should open (0 is the first/default camera). A tester with multiple
        cameras can set this per-machine via a ${CAMERA_INDEX} variable overridden in
        their local robot.toml, instead of editing the suite.
        '''
        return self.capture_and_predict(model_name,self.save_path,x=x,y=y,width=width,height=height,camera_index=camera_index)

    @keyword
    def capture_and_predict(self,
                          model_name,
                          save_path,
                          x = None,
                          y = None,
                          width = None,
                          height = None,
                          camera_index = 0):
        # If the caller didn't pin down every side of the crop box explicitly,
        # fall back to whatever ROI the model itself was trained with - and
        # only fall back further to the old arbitrary default if the model
        # has none (an older checkpoint, or one trained without annotate.py).
        if None in (x, y, width, height):
            roi = self.Base.get_capture_roi(model_name)
            if roi is None:
                logger.warn("No x/y/width/height given and this model has no stored "
                             "capture_roi - falling back to an arbitrary (100, 100, 200, 200) "
                             "box. Recalibrate with annotate.py --camera and retrain, or pass "
                             "x/y/width/height explicitly.")
                roi = {"x": 100, "y": 100, "width": 200, "height": 200}
            x = roi["x"] if x is None else x
            y = roi["y"] if y is None else y
            width = roi["width"] if width is None else width
            height = roi["height"] if height is None else height

        # Capture an image from the camera
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            raise RuntimeError(f"Error: camera (index {camera_index}) not accessible.")

        ret, frame = camera.read()
        camera.release()
        if not ret or frame is None:
            raise RuntimeError(f"Error: failed to read a frame from camera (index {camera_index}).")

        # Log the full frame with the crop region marked, so a miscalibrated
        # x/y/width/height is visible directly in log.html instead of only
        # showing up indirectly as a wrong/confusing prediction.
        marked_frame = frame.copy()
        cv2.rectangle(marked_frame, (x, y), (x + width, y + height), (0, 255, 0), 1)
        full_frame_path = str(Path(save_path).with_stem(Path(save_path).stem + "_full"))
        cv2.imwrite(full_frame_path, marked_frame)
        self._show_photo_in_log(full_frame_path)

        # Crop the ROI from the captured frame and classify only that crop
        roi_crop = frame[y:y+height, x:x+width]

        cv2.imwrite(save_path, roi_crop)
        self._show_photo_in_log(save_path)
        return self.Base._predict(model_name,save_path)
