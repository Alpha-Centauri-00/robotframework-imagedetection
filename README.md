# Robot Framework- ImageDetection
> Disclaimer: This "RobotFramework-ImageDetection" library is currently in early development and is intended for testing and experimentation purposes only. It is not recommended for use in production environments. Please use at your own risk. Feedback and bug reports are highly appreciated as we continue to improve the library. Thank you for your understanding.⚠️

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

4. **Split Data**: Ensure that each class's photos are split appropriately between the "training" and "validation" folders. The "training" folder should contain a majority of the photos for each class, while the "validation" folder should have a smaller subset of images for evaluation purposes.

Your data structure should look like this:

      Data/
      ├── training/
      │ ├── Right/
      │ ├── Left/
      │ └── Straight/
      └── validation/
      ├── Right/
      ├── Left/
      └── Straight/



Following this organized structure will enable smooth data loading and training using the "RobotFramework-ImageDetection" library. Remember to provide a sufficient amount of diverse and representative photos for accurate model training.

## Training and Detection

Once you have organized your data as described above, you can use the library to train your image detection model and perform real-time detection within your Robot Framework automation projects.

