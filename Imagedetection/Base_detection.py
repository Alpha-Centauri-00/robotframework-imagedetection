import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


__version__ = "0.0.7"

class Base_Detect:

    def __init__(self):
        self.epoch = 20

    def train_model(self,training_path, validation_path):
        
        class_num = len(self._get_subfolders("Data/training/"))

        train_datagen = ImageDataGenerator(rescale=1/255)
        valid_datagen = ImageDataGenerator(rescale=1/255)

        train_dataset = train_datagen.flow_from_directory(training_path,
                                                        target_size=(200, 200),
                                                        batch_size=3,
                                                        class_mode='categorical')

        valid_dataset = valid_datagen.flow_from_directory(validation_path,
                                                        target_size=(200, 200),
                                                        batch_size=3,
                                                        class_mode='categorical')

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(class_num, activation='softmax')  # 3 output neurons for 3 classes
        ])

        model.compile(  loss="categorical_crossentropy",  # Use categorical_crossentropy for multi-class classification
                        optimizer="adam",
                        metrics=["accuracy"])

        model.fit(  train_dataset,
                    steps_per_epoch=len(train_dataset),
                    epochs=self.epoch,
                    validation_data=valid_dataset,
                    validation_steps=len(valid_dataset))

        model.save(datetime.now().strftime(f'model_%Y%d%m%H%M.keras'))


    def _get_subfolders(self,path):
        subfolders = []
        try:

            for folder in os.listdir(path):
                folder_path = os.path.join(path,folder)
                if os.path.isdir(folder_path):
                    subfolders.append(folder)
            return subfolders
        except FileNotFoundError as e:
            print(e)

            
    #@tf.function
    def _predict(self,model_name, photo_path):
        model = load_model(model_name)
        # Load the testing image
        
        img = image.load_img(photo_path, target_size=(200, 200))
        X = image.img_to_array(img)
        X = np.expand_dims(X, axis=0)
        X /= 255.0  # Normalize the image
        prediction_probs = model.predict(X)[0]
        
        class_labels = self._get_subfolders("Data/training/")
        
        predicted_class = class_labels[np.argmax(prediction_probs)]
        # print("Predicted class:", predicted_class)
        # print("Predicted probs:", prediction_probs)
        return predicted_class
        #return prediction_probs

        

    def predict_from_google_model(self,_model,_image):
        from PIL import Image, ImageOps  
        
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Load the model
        model = load_model(_model, compile=False)

        # Load the labels
        class_names = open("labels.txt", "r").readlines()

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        image = Image.open(_image).convert("RGB")

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)


    