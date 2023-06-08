
<details>
  <summary><b>Code Explanatation Number Plate Recognition Using CNN</b></summary>
  
  
  ## Import Libraries: </br>
  ```python
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.optimizers import Adam
import xml.etree.ElementTree as ET
```
### Library Explanatation: </br>
 - <code>import os:</code> This library provides a way to interact with the operating system and access file paths and directories.
 - <code>import cv2:</code> This is the OpenCV library used for image processing and computer vision tasks.
 - <code>import numpy as np:</code> This imports the NumPy library for scientific computing and arrays.
 - <code>import tensorflow as tf:</code> This imports the TensorFlow library for machine learning and deep learning tasks.
 - <code>from tensorflow.keras.layers import Flatten, Dense:</code> This imports the Flatten and Dense layers from Keras, which are used to build neural networks.
 - <code>from tensorflow.keras.models import Model:</code> This imports the Model class from Keras, which is used to create a deep learning model.
 - <code>from tensorflow.keras.applications.vgg16 import VGG16:</code> This imports the pre-trained VGG16 model from Keras, which is a convolutional neural network commonly used for image classification tasks.
 - <code>from tensorflow.keras.applications.vgg19 import VGG19:</code> This imports the pre-trained VGG19 model from Keras, which is similar to VGG16 but has more layers.
 - <code>from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2:</code> This imports the pre-trained MobileNetV2 model from Keras, which is a lightweight convolutional neural network commonly used for mobile and embedded devices.
 - <code>from tensorflow.keras.optimizers import Adam:</code> This imports the Adam optimizer from Keras, which is an algorithm used to optimize the weights of a neural network during training.
 - <code>import xml.etree.ElementTree as ET:</code> This imports the ElementTree library for parsing XML files.
 
 
  ## Define Input Shape & Batch Size: </br>
```python
input_shape = (224, 224, 3)
batch_size = 32
```
### Code Explanatation: </br>
 - <strong>input_shape: </strong> This is a tuple that specifies the dimensions of the input images.In this case, the input images will have a width and height of 224 pixels and three color channels (red, green, and blue).
 -  <strong>batch_size: </strong> It is specifies the number of samples that will be fed into the model at once during training. In this case, the model will process 32 images at a time. 


  ## Define Base Model: </br>
```python
base_model = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
```
### Code Explanatation: </br>
This line of code creates a VGG16 model instance called base_model.
 - <strong> input_shape=input_shape: </strong> This parameter specifies the shape of the input data to the model.
 - <strong> weights='imagenet: </strong> TThis parameter specifies the pre-trained weights to use for the model.

  ## Custom Layer to the Pre-trained Model: </br>
```python
x = base_model.output
x = Flatten()(x)
x = Dense(4, activation='linear')(x)
model = Model(inputs=base_model.input, outputs=x)
```
### Code Explanatation: </br>
This code adds custom layers to the pre-trained model.
 - <strong> x = base_model.output: </strong> This line sets x to the output of the pre-trained VGG16 model, which is the last layer before the fully connected layers.
 - <strong> x = Flatten()(x): </strong> This line adds a Flatten layer to the model.
 - <strong> x = Dense(4, activation='linear')(x): </strong> This line adds a Dense layer to the model with 4 units and a linear activation function.
 - <strong> model = Model(inputs=base_model.input, outputs=x): </strong> This line creates a new model. This creates a new model that combines the pre-trained VGG16 model with our custom fully connected layers to perform object detection.
 


  ## Training configuration for the model: </br>
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
```
### Code Explanatation: </br>
This code sets up the optimizer, loss function, and evaluation metric for the model, preparing it for training.
 - <strong> optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001): </strong> This line creates an instance of the Adam optimizer with a learning rate of 0.0001. The optimizer is used during training to adjust the weights of the model in order to minimize the loss function.
 - <strong> model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy']): </strong> This line <code>loss='mse'</code> specifies that we will use mean squared error as the loss function during training, which measures the difference between the predicted output and the true output. <code>optimizer=optimizer</code> specifies that we will use the Adam optimizer instance created in the previous line. <code>metrics=['accuracy']</code> specifies that we will track the accuracy metric during training.
 


## Save the best model: </br>
```python
save_vgg16_model = tf.keras.callbacks.ModelCheckpoint(
    "/content/drive/MyDrive/Colab Notebooks/Number-Plate-Recognition-Model/vgg16model.h5", 
    monitor='accuracy', 
    save_best_only=True, 
    verbose=1
)
```
### Code Explanatation: </br>
This code snippet creates a callback function using the tf.keras.callbacks.ModelCheckpoint class that saves the best-performing model during the training process.

 - <code>"/content/drive/MyDrive/Colab Notebooks/Number-Plate-Recognition-Model/vgg16model.h5":</code> This parameter specifies the path where the model weights will be saved.
 - <strong> monitor='accuracy: </strong> This tells the function to monitor the model's accuracy during training.
 - <strong> save_best_only=True: </strong> This parameter ensures that only the best model (based on the monitored metric) will be saved. If set to False, the function will save the model after every epoch.
 - <strong> verbose=1: </strong> This parameter sets the verbosity level of the output messages during training. A value of 1 means that progress updates will be printed to the console.

## Directory Path: </br>
```python
training_directory = '/content/drive/MyDrive/Colab Notebooks/Number-Palte-Dataset/train'
validation_directory = '/content/drive/MyDrive/Colab Notebooks/Number-Palte-Dataset/valid'
```
### Code Explanatation: </br>
These are the directory paths for the training and validation datasets used in this project.


 
 
</details>
