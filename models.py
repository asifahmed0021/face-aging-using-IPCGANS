import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from keras import Input, Model
from keras.applications import InceptionResNetV2
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization
from keras.layers import Reshape, concatenate, LeakyReLU, Lambda
from keras import backend as K
from keras.layers import  Activation, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing import image
from models import *

def build_encoder():
  #defining parameters
  input_layer = Input(shape = (128, 128, 3))
  leaky_alpha=0.2

  # convolutional layer output shape(None,64,64,64)
  encoder = Conv2D(filters = 64, kernel_size = 5, strides = 2, padding = 'same')(input_layer)

  # normalisaion layer
  encoder = LeakyReLU(leaky_alpha)(encoder)
  
  # convolutional layer output shape(None,32,32,128)
  encoder = Conv2D(filters = 128, kernel_size = 5, strides = 2, padding = 'same')(encoder)

  # normalisation layer
  encoder = BatchNormalization()(encoder)
  encoder = LeakyReLU(leaky_alpha)(encoder)
  
  # convolutional layer output shape(None,16,16,256)
  encoder = Conv2D(filters = 256, kernel_size = 5, strides = 2, padding = 'same')(encoder)

  #normalisation layer
  encoder = BatchNormalization()(encoder)
  encoder = LeakyReLU(leaky_alpha)(encoder)
  
  # convolutional layer output shape(None,8,8,512)
  encoder = Conv2D(filters = 512, kernel_size = 5, strides = 2, padding = 'same')(encoder)

  #normalisation layer
  encoder = BatchNormalization()(encoder)
  encoder = LeakyReLU(leaky_alpha)(encoder)
  
  ## flattening layer output shape(None,8*8*512)
  encoder = Flatten()(encoder)
  
  ## reducing dimension and normalising output shape(None,4096)
  encoder = Dense(4096)(encoder)
  encoder = BatchNormalization()(encoder)
  encoder = LeakyReLU(leaky_alpha)(encoder)
  
  ## output layer shape(None,100)
  encoder = Dense(100)(encoder)
  
  
  ## create a model
  model = Model(inputs = [input_layer], outputs = [encoder])
  return model


def build_generator():
  # defining parameters
  latent_dims = 100
  num_classes = 12
  leaky_alpha=0.2

  # input latent vector and feature vector
  # shape(None,100)
  input_z_noise = Input(shape = (latent_dims, ))
  # shape(None,12)
  input_label = Input(shape = (num_classes, ))

  # concatenate both input vectors to make input vector for generatot of shape(None,112)
  x = concatenate([input_z_noise, input_label])
  
  # flattening and normalising
  x = Dense(2048, input_dim = latent_dims + num_classes)(x)
  x = LeakyReLU(leaky_alpha)(x)
  x = Dropout(leaky_alpha)(x)
  
  # Increasing dimension of current vector to make suitable for converting to a shape of(None,16,16,512)
  x = Dense(512 * 16 * 16)(x)
  x = BatchNormalization()(x)
  x = LeakyReLU(leaky_alpha)(x)
  x = Dropout(leaky_alpha)(x)
  
  # Reshaping to a shape of (None,16,16,512) 
  x = Reshape((16, 16, 512))(x)
  
  #  1st upsampling,convolution and normalisation
  x = UpSampling2D(size = (2, 2))(x)
  x = Conv2D(filters = 256, kernel_size = 5, padding = 'same')(x)
  x = BatchNormalization(momentum = 0.8)(x)
  x = LeakyReLU(leaky_alpha)(x)
  
  #  2nd upsampling,convolution and normalisation
  x = UpSampling2D(size = (2, 2))(x)
  x = Conv2D(filters = 128, kernel_size = 5, padding = 'same')(x)
  x = BatchNormalization(momentum = 0.8)(x)
  x = LeakyReLU(leaky_alpha)(x)
  
  #  3rd upsampling,convolution and normalisation
  x = UpSampling2D(size = (2, 2))(x)
  x = Conv2D(filters = 3, kernel_size = 5, padding = 'same')(x)
  x = Activation('tanh')(x)
  
  #final shape(None,128,128,3)
  
  model = Model(inputs = [input_z_noise, input_label], outputs = [x])
  return model


def build_discriminator():
  # defining parameters
  input_shape = (128, 128, 3)
  label_shape = (12, )
  image_input = Input(shape = input_shape)
  label_input = Input(shape = label_shape)
  leaky_alpha=0.2
  
  # convolution layer output shape (None,64,64,64)
  x = Conv2D(64, kernel_size = 3, strides = 2, padding = 'same')(image_input)

  # normalising
  x = LeakyReLU(leaky_alpha)(x)
  
  # expanding label input to make it suitable to concatenate with current convolution layer
  label_input1 = Lambda(expand_label_input)(label_input)

  # concatenating with current input layer
  # output shape(None,64,64,76)
  x = concatenate([x, label_input1], axis = 3)
  
  # convolution layer output shape(None,32,32,128)
  x = Conv2D(128, kernel_size = 3, strides = 2, padding = 'same')(x)

  # normalisation layer
  x = BatchNormalization()(x)
  x = LeakyReLU(leaky_alpha)(x)
  
  # convolution layer output shape(None,16,16,256)
  x = Conv2D(256, kernel_size = 3, strides = 2, padding = 'same')(x)

  # normalisation layer
  x = BatchNormalization()(x)
  x = LeakyReLU(leaky_alpha)(x)
  
  #convolution layer output shape(None,8,8,512)
  x = Conv2D(512, kernel_size = 3, strides = 2, padding = 'same')(x)

  # normalisation layer
  x = BatchNormalization()(x)
  x = LeakyReLU(leaky_alpha)(x)
  
  # flattening layer output shape(None,8*8*512)
  x = Flatten()(x)
  # final single output shape (None,1)
  x = Dense(1, activation = 'sigmoid')(x)
  
  
  model = Model(inputs = [image_input, label_input], outputs = [x])
  return model


def build_fr_model(input_shape):
  
  resnet_model = InceptionResNetV2(include_top = False, weights = 'imagenet',
                                   input_shape = input_shape, pooling = 'avg')
  image_input = resnet_model.input
  x = resnet_model.layers[-1].output
  out = Dense(128)(x)
  embedder_model = Model(inputs = [image_input], outputs = [out])
  
  input_layer = Input(shape = input_shape)
  
  x = embedder_model(input_layer)
  output = Lambda(lambda x: K.l2_normalize(x, axis = -1))(x)
  
  
  model = Model(inputs = [input_layer], outputs = [output])
  return model