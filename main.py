from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
from keras.models import load_model
import cv2
import os
from robot.api import logger

def train_model(training_path, validation_path):
    
    class_num = len(_get_subfolders("Data/training/"))

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
                epochs=20,
                validation_data=valid_dataset,
                validation_steps=len(valid_dataset))

    model.save("model.keras")


def _get_subfolders(path):
    subfolders = []
    try:

        for folder in os.listdir(path):
            folder_path = os.path.join(path,folder)
            if os.path.isdir(folder_path):
                subfolders.append(folder)
        return subfolders
    except FileNotFoundError as e:
        print(e)

def capture_and_predict(model_name):
    # Take a photos from a camera

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: camera not accessible.")
        return None
    
    _, frame = camera.read()

    # Provide the desired save path for the captured photo
    save_path = "captured_photo2.jpg"
    cv2.imwrite(save_path, frame)
    camera.release()

    _predict(model_name,save_path)
        
    curdir = os.getcwd()
    current_directory_with_double_backslashes = curdir.replace("\\", "\\\\")

    Log_photo = fr"<img src={current_directory_with_double_backslashes}\\{save_path} width='150' height='100'>"
    logger.info(Log_photo,html=True,also_console=False)


def _predict(model_name, photo):
    model = load_model(model_name)
    # Load the testing image
    
    img = image.load_img(photo, target_size=(200, 200))
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    X /= 255.0  # Normalize the image
    prediction_probs = model.predict(X)[0]
    
    class_labels = _get_subfolders("Data/training/")
    
    predicted_class = class_labels[np.argmax(prediction_probs)]
    print("Predicted class:", predicted_class)
    #print("Predicted probs:", prediction_probs)


def predict_from_path(model_name, photo_path):
    # You need to train a model first to use this function. 
    
    _predict(model_name,photo_path)

    Log_photo = f'<img src="{photo_path}" width="150" height="100">'
    logger.info(Log_photo,html=True,also_console=False)


