# Robot Framework- ImageDetection
> Disclaimer: This library is currently in early development and is intended for testing and experimentation purposes only. It is not recommended for use in production environments. Please use at your own risk. Feedback and bug reports are highly appreciated as we continue to improve the library. Thank you for your understanding.⚠️

The "RobotFramework-ImageDetection" library is a powerful tool designed to harness the capabilities of machine learning to train and detect photos effectively within the Robot Framework automation framework. With its seamless integration, this library enables users to build intelligent and sophisticated automation solutions by incorporating computer vision and image recognition techniques.


## Installation

Before installing "RobotFramework-ImageDetection," please ensure you have Python 3.6 or higher installed on your system.

### Using pip

Setting up a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Windows, use: venv\Scripts\activate
```

To install the library, simply use pip:

```bash
pip install robotframework-imagedetection
```

## Collecting Dataset / Photos

Before training the image detection model with the "RobotFramework-ImageDetection" library, you need to collect and organize your data/photos in a specific way. Follow these steps to set up your data:

>  &#x1F4A1; For your convenience, we have provided a sample dataset in another repository, which you can clone and use for testing purposes. You can find the sample dataset at [Data-Example](https://github.com/Alpha-Centauri-00/Data-Example/tree/main). Clone the repository and follow the same folder structure to get started with the "RobotFramework-ImageDetection" library.

***

1. **Create the Main Dataset Folder**: Start by creating a new folder in your project called "Data." This folder will serve as the main directory for your training and validation data.

2. **Create Training and Validation Subfolders**: Inside the "Data" folder, create two subfolders: "training" and "validation." These folders will contain your training and validation datasets, respectively.

3. **Organize Photos by Class**: Within the "training" and "validation" folders, organize your photos into subfolders based on their classes or categories. For example, if you are classifying images into "Right," "Left," and "Straight," create subfolders named "Right," "Left," and "Straight" inside both the "training" and "validation" folders.

4. **Split Data**: Ensure that each class's photos are distributed appropriately between the "training" and "validation" folders. The "training" folder should contain photos of each class. Similarly, the "validation" folder should also contain photos of each class, but make sure to use different images than those present in the "training" folder. This separation allows for proper evaluation and testing of the image detection model.

Your data structure should look like this:

      Data/
      ├── training/
      │ ├── Right/
      │ ├── Left/
      │ └── Straight/
      ├── validation/
      │ ├── Right/
      │ ├── Left/
      │ └── Straight/



Following this organized structure will enable smooth data loading and training using the "RobotFramework-ImageDetection" library. Remember to provide a sufficient amount of diverse and representative photos for accurate model training.

## Train and Detect

### Training a new model
Once you have organized your data as described above, you can use the library to train your image detection model and perform real-time detection within your Robot Framework automation projects.


Now create a new Test case using Robot framework to train a new Model:

```robot
*** Settings ***
Library   Imagedetection

*** Variables ***
${training_folder}      ${CURDIR}\\Data\\training
${validation_folder}    ${CURDIR}\\Data\\validation

*** Test Cases ***

Training a New Model
    Train Model    ${training_folder}    ${validation_folder}
```


In this example, the Robot Framework test case named "Training a New Model" starts by importing the "Imagedetection" library under setting section.

The Variables section sets up two variables:

***${training_}***: Specifies the directory path containing the training data for the model. It points to the "training" folder inside the "Data" directory.
***${validati_}***: Specifies the directory path containing the validation data for the model. It points to the "validation" folder inside the "Data" directory.
The actual test case named "Training a New Model" calls the "Train Model" keyword from the "Imagedetection" library. This keyword is designed to train an image detection model using the specified training and validation data directories.

> Training a model can take some time, depends of course on the size of your dataset. Please be patient 🙂

Once you have set up your test cases like this example, you can run your Robot Framework tests to train and utilize your image detection model effectively.

after the test is finished, it will automaticlly generate a new file called `model.keras` This file will contain the trained model, which can be used for image detection in your subsequent Robot Framework test cases. This model file is crucial, as it encapsulates the learned patterns and features from the training data, allowing it to accurately detect objects or patterns in new images. Once the file is generated, you can load and utilize the trained model to perform image detection tasks with ease.

***
### Detecting from new model

Now, the easy part is to use another keyword `Detect From Path` to make predictions using the newly trained model. This keyword takes two arguments: `model_name`, which is the path to the generated model, and `photo_path`, which indicates the path to a test photo that will be used to check if our model can detect the object or not. By providing these two arguments, you can easily evaluate the performance of your trained model on new images and test its detection capabilities.

```robot
*** Settings ***
Library   Imagedetection

*** Variables ***
${training_folder}      ${CURDIR}\\Data\\training
${validation_folder}    ${CURDIR}\\Data\\validation

${model_name}           ${CURDIR}\\model_XXXXX.keras

*** Test Cases ***

Check the test photo
    Detect From Path    ${model_name}    ${CURDIR}\\test\\Left_test2png.png
```
`${model_name}`: Specifies the path to the generated model file (model.keras) that was trained using the specified training and validation data directories.
This test case "Check the test photo" calls the "Detect From Path" keyword to perform image detection on a single test photo using the trained model specified.

## Conclusion

Congratulations! You've reached the end of the README for the "RobotFramework-ImageDetection" library. We hope this documentation has provided you with a clear understanding of how to use our library for image detection in your Robot Framework projects.

In this README, we covered the following topics:

- Introduction to the library and its features
- How to collect and organize your dataset
- Training a new model and generating the `model.keras` file
- Using the trained model for image detection with `Detect From Path`

Our library aims to simplify image detection tasks and empower you to build robust and accurate image detection systems within the Robot Framework.

If you encounter any issues, have questions, or want to contribute to the project, feel free to visit our [GitHub repository](https://github.com/Alpha-Centauri-00/robotframework-imagedetection). We value your feedback and are excited to grow the library together with the community.

Thank you for choosing "RobotFramework-ImageDetection" for your image detection needs. Happy testing and happy detecting!😄


