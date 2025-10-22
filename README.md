Emotion Detector using Webcam

This project detects human emotions in real time using a webcam.
It is based on a convolutional neural network (CNN) model trained with the FER2013 dataset.

The goal is to identify basic facial expressions such as happiness, sadness, anger, surprise, and neutrality.
It uses the webcam to capture live images, processes them, and displays the detected emotion on the screen.

How it works

The FER2013 dataset is used to train a CNN model for emotion classification.

The model is saved as emotion_detector_model.h5.

OpenCV is used to access the webcam and detect faces in real time.

Each detected face is analyzed by the trained model to predict the emotion.

The result is shown directly on the webcam feed.

Technologies used

Python

TensorFlow / Keras

OpenCV

FER2013 dataset

How to run the project

Clone the repository or download the files.

Make sure you have Python and the required libraries installed.

Run the following command:

python detect_emotion_live.py


The webcam window will open and display the detected emotions in real time.

!!### Dataset
This project uses the FER-2013 dataset for training.  
Due to size limitations, it is **not included** in the repository.  

You can download it here: [FER-2013 Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)  
After downloading, extract it into the folder structure:

fer2013/
├── train/
├── test/



Possible improvements

Add more emotion categories

Improve model accuracy with more data


Create a simple user interface
