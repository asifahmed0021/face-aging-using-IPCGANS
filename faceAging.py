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
from utils import *

if __name__ == '__main__':
  ##Note:In comments 'None' refers to dataset size.

  #defining hyper parameters
  directory_name='wiki_crop'
  image_shape=(128,128,3)
  latent_vector_size=100
  


  #building and compiling discriminator model
  discriminator = build_discriminator()
  dis_optimizer = Adam(lr = 0.0002, beta_1 = 0.5, beta_2 = 0.999, epsilon = 10e-8)
  discriminator.compile(loss = ['binary_crossentropy'],optimizer = dis_optimizer)



  #building and compiling generator model
  generator = build_generator()
  gen_optimizer = Adam(lr = 0.0002, beta_1 = 0.5, beta_2 = 0.999, epsilon = 10e-8)
  generator.compile(loss = ['binary_crossentropy'],optimizer = gen_optimizer)



  ##building and compiling adversarial model(generator+discriminator)
  discriminator.trainable = False
  #building fake input latent vector and fake input label to make adversarial model
  fake_input_latent = Input(shape = (100, ))
  fake_input_label = Input(shape = (12, ))

  # taking fake output from generator
  fake_gen_output = generator([fake_input_latent, fake_input_label])

  #taking discriminator output for these fake_gen_output
  fake_dis_output = discriminator([fake_gen_output, fake_input_label])

  #making adversarial model
  adversarial_model = Model(inputs = [fake_input_latent, fake_input_label],outputs = [fake_dis_output])
  adversarial_optimizer = Adam(lr = 0.0002, beta_1 = 0.5, beta_2 = 0.999, epsilon = 10e-8)
  adversarial_model.compile(loss = ['binary_crossentropy'],optimizer = gen_optimizer)



  #loading the dataset
  image_list,[gender_list,age_list]=load_data(directory_name,'wiki') 
  #temporary
  image_list=image_list[:111]
  gender_list=gender_list[:111]
  age_list=age_list[:111]
  cat_age_list=age_to_category(age_list) 
  categorical_label=generate_label(cat_age_list,gender_list) #shape(None,12)
  final_feature = np.reshape(np.array(categorical_label), [len(categorical_label), 12]) #final real categorical feature shape(None,12)
  real_loaded_images=load_images(image_list,directory_name,(128,128,3)) #final real loaded images shape(None,128,128,3)
  #hyper parameters for training
  dataset_size=len(image_list)
  epochs=200
  batch_size=10
  num_of_batches=int(dataset_size/batch_size)

  #real and fake labels for discriminators output
  real_labels_batch = np.ones((batch_size, 1), dtype = np.float32)
  fake_labels_batch = np.zeros((batch_size, 1), dtype = np.float32)



  ##Training discriminator and generator.............................................................
  gen_loss_track=[]
  dis_loss_track=[]
  for epoch in range(epochs):
    print("Epoch: {}".format(epoch+1)+"................................................................................")
    for batch in range(num_of_batches):
      print("Batch: {}".format(batch + 1) +" of Epoch:{}".format(epoch+1))
      #getting images for current batch and normalizing
      real_images_batch = real_loaded_images[batch * batch_size:(batch + 1) * batch_size]
      real_images_batch = real_images_batch / 127.5 - 1.0
      real_rimages_batch = real_images_batch.astype(np.float32)
      #getting feature for current batch
      real_feature_batch = final_feature[batch * batch_size: (batch + 1) * batch_size]
      #making fake latent vector
      fake_latent = np.random.normal(0, 1, size = (batch_size, latent_vector_size))


      ##training discriminator
      gen_fake_output=generator.predict_on_batch([fake_latent,real_feature_batch])
      d_loss_real = discriminator.train_on_batch([real_images_batch,real_feature_batch], real_labels_batch)
      d_loss_fake = discriminator.train_on_batch([gen_fake_output,real_feature_batch], fake_labels_batch)
      dis_loss_track.append((d_loss_real+d_loss_fake)/2)


      ##training generator
      fake_latent_2 = np.random.normal(0, 1, size = (batch_size, latent_vector_size))
      fake_feature_batch = np.random.randint(0, 12, batch_size).reshape(-1, 1)
      fake_feature_batch = to_categorical(fake_feature_batch, 12)   
      g_loss = adversarial_model.train_on_batch([fake_latent_2, fake_feature_batch],real_labels_batch)
      gen_loss_track.append(g_loss)

      #printing loss for current batch in epoch
      print("Current generator loss:{}".format(g_loss))
      print("Current discriminator loss:{}".format((d_loss_real+d_loss_fake)/2))
    

    #generate images after every 5th epoch and save them
    sample_feature=final_feature[:5]
    sample_latent=np.random.normal(0, 1, size = (5,latent_vector_size))
    sample_gen_images=generator.predict_on_batch([sample_latent,sample_feature])
    for i, img in enumerate(sample_gen_images[:5]):
      save_rgb_img(img, path = "results/img_{}_{}.png".format(epoch, i))


  #saving model weights in files
  generator.save_weights("generator.h5")
  discriminator.save_weights("discriminator.h5")




  ##training encoder..................................................
  encoder = build_encoder()
  encoder.compile(loss = euclidean_distance_loss,optimizer = 'adam')
  generator.load_weights("generator.h5")
  #making fake latent and feature vector for encoder
  fake_latent = np.random.normal(0, 1, size = (5000,latent_vector_size))
  fake_feature = np.random.randint(low = 0, high = 12, size = (5000, ),dtype = np.int64)
  fake_feature = np.reshape(np.array(fake_feature), [len(fake_feature), 1])
  fake_feature = to_categorical(fake_feature, num_classes = 12)

  enc_loss_track=[]
  for epoch in range(epochs):
    print("Epoch: {}".format(epoch))
    no_batches=5000/batch_size
    for batch in range(no_batches):
      print("Batch: {}".format(batch))
      #from fake latent and feature vector taking batch size of them
      fake_latent_batch = fake_latent[batch * batch_size: (batch + 1) * batch_size]
      fake_feature_batch = fake_feature[batch * batch_size: (batch + 1) * batch_size]
      generated_images = generator.predict_on_batch([fake_latent_batch,fake_feature_batch])
      #training
      encoder_loss = encoder.train_on_batch(generated_images,fake_latent_batch)
      print("Encoder loss: ", encoder_loss)
      enc_loss_track.append(encoder_loss)
  encoder.save_weights("encoder.h5")

  ##optimising generator and encoder by identity preservation
  # Load the encoder network
  encoder = build_encoder()
  encoder.load_weights("encoder.h5")
  ## Load the generator network
  generator.load_weights("generator.h5")
  #build and compile fr model
  fr_model = build_fr_model((128,128,3))
  fr_model.compile(loss = ['binary_crossentropy'],optimizer = 'adam')
  fr_model.trainable = False

  #input layers
  input_image = Input(shape = (128, 128, 3))
  input_label = Input(shape = (12, ))
  #use the encoder and the generator network
  latent0 = encoder(input_image)
  gen_images = generator([latent0, input_label]) 
  embeddings = fr_model(gen_images)

  #create a model
  fr_adversarial_model = Model(inputs = [input_image, input_label],outputs = [embeddings])
  #compile the model
  fr_adversarial_model.compile(loss = euclidean_distance_loss,optimizer = adversarial_optimizer)

  ##training
  recons_loss_track=[]
  for epoch in range(epochs):
    print("Epoch: {}".format(epoch+1)+"................................................................................")
    for batch in range(num_of_batches):
      print("Batch: {}".format(batch + 1) +" of Epoch:{}".format(epoch+1))
      #real images for batch
      images_batch = real_loaded_images[batch * batch_size: (batch + 1) * batch_size]
      images_batch = images_batch / 127.5 - 1.0
      images_batch = images_batch.astype(np.float32)
      #real feature for batch
      feature_batch = final_feature[batch * batch_size: (batch + 1) * batch_size]
      real_embeddings = fr_model.predict_on_batch(images_batch)
      #train
      recons_loss = fr_adversarial_model.train_on_batch([images_batch, feature_batch], real_embeddings)
      recons_loss_track.append(recons_loss)
      print("Reconstruction loss: ", recons_loss)

    if epoch % 10 == 0:
      images_batch = real_loaded_images[0:batch_size]
      images_batch = images_batch / 127.5 - 1.0
      images_batch = images_batch.astype(np.float32)

      feature_batch = final_feature[0:batch_size]
      latent_batch =encoder.predict(images_batch)
        
      gen_images = generator.predict_on_batch([latent_batch, feature_batch])
        
      for i, img in enumerate(gen_images[:5]):
        save_rgb_img(img, path = "results/img_opt_{}_{}.png".format(epoch, i))
  generator.save_weights("generator_optimized.h5")
  encoder.save_weights("encoder_optimized.h5")


