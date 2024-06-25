# ws2022-group-3

## Name
Fritzi - A virtual interactive 3D companion

## Description
Fritzi is an interactive 3D virtual companion with the ability to communicate, detect and classify objects specified by the user. Therefore, a combination of different techniques in the areas of machine learning, computer vision, voice interaction and web modelling are used. Images can be provided by an upload function or via webcam feed. After starting the webcam, the objects in the feed are localized and marked with bounding boxes. The resulting object for classification can be selected by fingerpointing on a bounding box using your index fingertip. Besides classifying the object Fritzi allows the user to add and show synonym labels, new labels and explain the result of the prediction. Moreover, it is possible to use voice interaction and retrain the model when new labels are added. Therefore, new images for the training process can be downloaded from the internet.
For further information see: https://hcai.eu/iml/?p=1191&preview=true

## Installation
1) Clone this repository and open it in the Pycharm IDE
2) Install the listed Requirements using the given statements for both Node.js and Python
3) Start the project by running the cell in the jupyter notebook gui.ipynb (automatically starts a jupyter server)
4) Wait until the server was started
5) Open http://localhost:3000 in your browser to load Fritzi
6) The application setup is ready to be used, interact and have fun with Fritzi!

## Requirements
### Python
- Conda  create -n [env_name]
- Conda activate [env_name]
- conda install python=3.9.12
- pip install traitlets
- pip install opencv-python
- pip install lime
- pip install tensorflow
- pip install albumentations
- pip install jupyter
- pip install boltons
- pip install SpeechRecognition
- pip install pyaudio
- pip install mediapipe
- pip install onnxruntime –user
- pip install onnx --user

### Node.js
- install Node.js and npm (node package manager) on your computer
- navigate to the node folder containing the index.js and index.html and open the command line
- run npm install to install all needed dependencies from the package.json file
- the avatar is ready now
