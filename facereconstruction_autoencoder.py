# -*- coding: utf-8 -*-
"""
# Introduction

We trying to reconstruct the masked face with the help of auto endcoder. Here we have defined the architecture of the auto endcoder we have used.

## Import Necessary Libraries
"""

import tensorflow as tf
import keras 
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Dropout, Input
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm 
import numpy as np
import os
import re

"""## Load data
Here we  are going to load images each with mask and no mask. These images are converted to an array and are appended in empty array. Here we also have defind function to load data serially. 
"""

def sorted_alphanumeric(data):  
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key = alphanum_key)

SIZE = 256

mask_path = '../input/face-mask-lite-dataset/with_mask'
mask_array = []

image_path = '../input/face-mask-lite-dataset/without_mask'
img_array = []

image_file = sorted_alphanumeric(os.listdir(image_path))
mask_file = sorted_alphanumeric(os.listdir(mask_path))
for i in tqdm(mask_file):
    if i == 'with-mask-default-mask-seed2500.png':
        break
    else:    
        image = cv2.imread(mask_path + '/' + i,1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (SIZE, SIZE))
        image = image.astype('float32') / 255.0    
        mask_array.append(img_to_array(image))
    
for i in tqdm(image_file):
 
    if i == 'seed2500.png':
        break
    
    else:
        image = cv2.imread(image_path + '/' + i,1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (SIZE, SIZE))
        image = image.astype('float32') / 255.0
        img_array.append(img_to_array(image))

"""## Plot image pair"""

def plot_image_pair(images = 5):
    for i in range(images):
        plt.figure(figsize = (7,7))
        plt.subplot(1,2,1)
        plt.title("No Mask", fontsize = 15)
        plt.imshow(img_array[i].reshape(SIZE, SIZE, 3))
        plt.subplot(1,2,2)
        plt.title("Mask", fontsize = 15)
        plt.imshow(mask_array[i].reshape(SIZE, SIZE, 3)) 
                
plot_image_pair(5)

"""## Slicing and reshaping
Here we have used 75% images for training and remaining 25% for testing.
"""

train_mask_image = mask_array[:2300]
train_image = img_array[:2300]
test_mask_image = mask_array[2300:]
test_image = img_array[2300:]
train_mask_image = np.reshape(train_mask_image,(len(train_mask_image),SIZE,SIZE,3))
train_image = np.reshape(train_image, (len(train_image),SIZE,SIZE,3))
print('Train no mask image shape:',train_image.shape)
test_mask_image = np.reshape(test_mask_image,(len(test_mask_image),SIZE,SIZE,3))
test_image = np.reshape(test_image, (len(test_image),SIZE,SIZE,3))
print('Test no mask image shape',test_image.shape)

"""## Defining our model

Here we have used Conv2D and MaxPool2D in encoder network for downsampling. Latent vector is of shape (16,16,64). This latent vector is input for decoder network, decoder network tries to reconstruct images and tries to reduce reconstruction loss by upsampling this latent vector.
"""

encoder_input = keras.Input(shape=(SIZE,SIZE, 3), name="img")
x = Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', padding = 'same')(encoder_input)
x = MaxPool2D(pool_size = (2,2))(x)
x = Conv2D(filters = 32,kernel_size = (3,3),strides = (2,2), padding = 'valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)
x = Conv2D(filters = 64, kernel_size = (3,3), strides = (2,2), activation = 'relu', padding = 'same')(x)
x = MaxPool2D(pool_size = (2,2))(x)
x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)
x = Conv2D(filters = 128, kernel_size = (3,3), padding = 'same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)
x = Conv2D(filters = 128 , kernel_size = (3,3), activation = 'relu', padding = 'same')(x) 
x = Conv2D(filters = 256 , kernel_size = (3,3), padding = 'same')(x) 
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)
encoder_output = Conv2D(filters = 512 , kernel_size = (3,3), activation = 'relu', padding = 'same')(x) 
encoder = tf.keras.Model(encoder_input, encoder_output)



decoder_input = Conv2D(filters = 512 ,kernel_size = (3,3), activation = 'relu', padding = 'same')(encoder_output)
x = UpSampling2D(size = (2,2))(decoder_input)
x = Conv2D(filters = 256, kernel_size = (3,3),  padding = 'same')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.LeakyReLU()(x)
x = Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
x = Conv2D(filters = 128, kernel_size = (3,3),  padding = 'same')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.LeakyReLU()(x)
x = Conv2D(filters = 164, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
x = UpSampling2D(size = (2,2) )(x)

x = Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
x = UpSampling2D(size = (2,2) )(x)
x = Conv2D(filters = 32 , kernel_size = (3,3),  padding = 'same')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.LeakyReLU()(x)
x = UpSampling2D(size = (2,2) )(x) 
x = Conv2D(filters = 16  , kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
decoder_output = Conv2D(filters = 3, kernel_size = (3,3), padding = 'same')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.LeakyReLU()(x)

# final model
model = keras.Model(encoder_input, decoder_output)
model.summary()

"""## Compiling our model"""

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mean_absolute_error',
              metrics = ['acc'])

model.fit(train_mask_image, train_image, epochs = 100, verbose = 0)

"""## Model evaluation"""

loss_acc= model.evaluate(test_mask_image, test_image)
print("Loss: ",loss_acc[0])
print('Accuracy: ', np.round(loss_acc[1],2) * 100)

"""## plotting images"""

def plot_images(start = 0, end = 5):
    for i in range(start, end, 1):
        plt.figure(figsize = (10,10))
        plt.subplot(1,3,1)
        plt.title("No Mask", fontsize = 12)
        plt.imshow(test_image[i])
        plt.subplot(1,3,2)
        plt.title("Mask", fontsize = 12)
        plt.imshow(test_mask_image[i])
        plt.subplot(1,3,3)
        plt.title("Predicted", fontsize = 12)
        prediction = model.predict(test_mask_image[i].reshape(1,SIZE, SIZE, 3)).reshape(SIZE, SIZE, 3)
        plt.imshow(prediction)
        plt.show()

plot_images(20,30)