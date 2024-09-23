# Patient Fall Detection and Prevention

## Overview
This project aims to provide a real-time solution for detecting and preventing patient falls in healthcare settings. Using deep learning approaches and pose estimation, the system monitors and alerts healthcare staff when a fall is detected. The project is built using Python 3.10 and the PyTorch framework and is designed to run on Windows, with support for both standalone and server deployment. Pose estimation approach used to find the key-points for the patient. 

## Features
- `Real-Time Patient Detection`: Continuously monitors patients to detect falls in real-time.
- `Pose Estimation`: Leverages pose estimation models to analyze patient movements and identify potential fall risks.
- `Deep Learning`: Utilizes state-of-the-art deep learning techniques to improve accuracy.
- `Windows Installer`: Available as an executable for easy installation on Windows systems.
- `Server Deployment`: Supports deployment on servers for use in larger healthcare networks.

## Cases
`case 1:` Low Risk <br>
`case 2:` Moderate <br>
`case 3:` Highest Alert <br>
`case 4:` Emergency <br>

## Requirements
Python 3.10 <br>
PyTorch <br>
OpenCV <br>
Numpy <br>
Other dependencies listed in `requirements.txt`

## Installation
1- Clone this repository:

```shell
git clone https://github.com/javaidiqbal11/Patient_Fall_Detection.git
cd patient-fall-detection
```
2. Install the required packages:
```shell
pip install -r requirements.txt
```
3. Ensure you have the necessary hardware for running deep learning models (e.g., GPU if required).

## Running the Fall Detection System
To start the fall detection system:

```shell
python fall_detector.py
```
This will launch the application and start monitoring patient movements in real-time.

## Deployment
**Standalone (Windows Installer)**
- Download the pre-built Windows installer.
- Follow the installation instructions to set up the application on your system.
- Run the application directly from the installed program.

## Server Deployment
1. Configure your server environment.
2. Deploy the fall detection system on your server by setting up the required dependencies and running the system as a background process or service.
3. Ensure the network is configured to handle real-time data processing.

## Model Training and Fine-tuning
This project uses pre-trained pose estimation models. If you need to train or fine-tune the model:

1. Place your dataset in the appropriate folder.
2. Modify the training script as needed.
3. Run the training command:

```shell
python algorithms.py
```

## Importent Points 
Here are the short and precise points:

- `Pose Estimation`: OpenPose-based pose estimation technique.
- `Multipose or Single Pose`: Multi-pose estimation for detecting multiple persons.
- `Standing or Walking Identification`: Uses key body points (legs, torso) to identify standing or walking postures.
- `Algorithm Used`: OpenPose algorithm for real-time pose detection.
- `Sample Keypoints`: Nose, neck, shoulders, elbows, wrists, hips, knees, and ankles.
- `Deep Learning Algorithm`: Convolutional Neural Networks (CNN) used in OpenPose.
- `Dataset`: Custom dataset or open datasets related to human pose and activity recognition (likely using datasets like COCO).

## Contributions
We welcome contributions! Please follow the standard GitHub flow for submitting pull requests.
