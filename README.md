# Railway Crack Detection Bot

This project involves a railway crack detection bot designed to identify and report cracks in railway tracks in real-time. The dataset for this project was manually obtained, and the model has been trained using this data.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Research Paper](#research-paper)
- [Installation](#installation) 

## Introduction
Railway crack detection is a crucial task for ensuring the safety and reliability of railway transportation. This project leverages machine learning techniques to detect cracks on railway tracks using images.

## Dataset
The dataset used for training the model was manually obtained. It includes images of railway tracks with and without cracks. Proper labeling and preprocessing steps were performed to ensure the dataset's quality.

### Research Paper
For more detailed information about the methodology and results, you can read the research paper [here](https://dx.doi.org/10.2139/ssrn.4590664).

### Installation

1. Clone the repository:
     ```sh
     git clone https://github.com/harshalsoman/Crack-Detection-Bot.git
   
2. Running the Model
  The model used in this project is a convolutional neural network (CNN) that has been trained on the manually obtained dataset. The training process and model architecture details are available in the model.py file. To train or test the model, use the model.py script:
    ```sh
    python model.py

3. Real-time Detection
The real-time detection script, realtime.py, uses the trained model to detect cracks in real-time. Ensure that your camera or video input is correctly configured before running this script. For real-time crack detection, use the realtime.py script:
    ```sh
    python realtime.py
    
