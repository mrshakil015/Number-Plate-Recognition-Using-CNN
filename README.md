
<details>
  <summary><b>VGG16 MODEL</code> problem</b></summary>
  <code>pyresparser</code> is a simple resume parser used for extracting information from resumes. pyresparser work with <code>spacy</code>. But now it don't work properly in <code>spacy latest version.</code> It's work better in <code>spacy==2.3.8</code> When we run pyresparser in <code>spacy</code>latest version show <code>config.cfg</code> problem. To solve this problem <code>create virtual environment.</code> 

  
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
### Library Explanation: </br>
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
### Code Explanation: </br>
 - <strong>input_shape: </strong> This is a tuple that specifies the dimensions of the input images.In this case, the input images will have a width and height of 224 pixels and three color channels (red, green, and blue).
 -  <strong>batch_size: </strong> It is specifies the number of samples that will be fed into the model at once during training. In this case, the model will process 32 images at a time. 


  ## Define Base Model: </br>
```python
base_model = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
```
### Code Explanation: </br>
This line of code creates a VGG16 model instance called base_model.
 - <strong> input_shape=input_shape: </strong> This parameter specifies the shape of the input data to the model.
 - <strong> weights='imagenet: </strong> TThis parameter specifies the pre-trained weights to use for the model.



</details>
