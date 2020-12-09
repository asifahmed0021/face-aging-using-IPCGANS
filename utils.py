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

from scipy.io import loadmat
#function to load all image paths and image labels
def load_data(wiki_dir, dataset = 'wiki'):
  ## Loading the wiki.mat file
  meta = loadmat(os.path.join(wiki_dir, "{}.mat".format(dataset)))
  
  ## Load the list of all files
  full_path = meta[dataset][0, 0]["full_path"][0]
  
  ## List of Matlab serial date numbers
  dob = meta[dataset][0, 0]["dob"][0]
  
  #list of corresponding genders
  gender=meta[dataset][0,0]["gender"][0]

  ## List of years when photo was taken
  photo_taken = meta[dataset][0, 0]["photo_taken"][0]  # year
  
  ## Calculate age for all dobs
  age = [calculate_age(photo_taken[i], dob[i]) for i in range(len(dob))]
  
  ## Create a list of tuples containing a pair of an image path and age
  images = []
  age_list = []
  for index, image_path in enumerate(full_path):
    images.append(image_path[0])
    age_list.append(age[index])
  
  ## Return a list of all images and respective age
  return images, [gender,age_list]



from datetime import datetime

def calculate_age(photo_taken_date,birth_date):
  birth = datetime.fromordinal(max(int(birth_date) - 366, 1))
  
  if birth.month < 7:
    return photo_taken_date - birth.year
  else:
    return photo_taken_date - birth.year - 1


def euclidean_distance_loss(y_true, y_pred):
  
  """
  Euclidean distance
  https://en.wikipedia.org/wiki/Euclidean_distance
  y_true = TF / Theano tensor
  y_pred = TF / Theano tensor of the same shape as y_true
  returns float
  """
  
  return K.sqrt(K.sum(K.square(y_pred - y_true), axis = -1))


def save_rgb_img(img, path):
  
  """
  Save an RGB image
  """
  
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.imshow(img)
  ax.axis("off")
  ax.set_title("Image")
  
  plt.savefig(path)
  plt.close()

def age_to_category(age_list):
  
  cat_list = []
  
  for age in age_list:
    if 0 < age <= 18:
      cat = 0
    elif 18 < age <= 29:
      cat = 1
    elif 29 < age <= 39:
      cat = 2
    elif 39 < age <= 49:
      cat = 3
    elif 49 < age <= 59:
      cat = 4
    elif age >= 60:
      cat = 5
      
    cat_list.append(cat)
    
  return cat_list

def expand_label_input(x):
  x = K.expand_dims(x, axis = 1)
  x = K.expand_dims(x, axis = 1)
  x = K.tile(x, [1, 64, 64, 1])
  return x

#function to load all images as numpy array
def load_images(image_list,data_dir,image_shape):
  images=None
  for image_path in image_list:
    #loading single image 
    loaded_image=cv2.imread(os.path.join(data_dir,image_path))
    #converting to rgb 
    converted_rgb_image=cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
    #resizing image
    resized_rgb_image=cv2.resize(converted_rgb_image,(128,128))
    #expanding dimension for sample size
    final_image = np.expand_dims(resized_rgb_image, axis = 0)
    #append to images list
    if images is None:
        images = final_image
    else:
        images = np.concatenate([images,final_image], axis = 0)
  return images


def generate_label(age_label_list,gender_label_list):
  feature=[]
  for i in range(len(age_label_list)):
    if(gender_label_list[i]==1):
      feature.append(age_label_list[i])
    else:
      feature.append(age_label_list[i]+6)
  feature=to_categorical(feature,12)
  return feature
