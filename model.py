 
get_ipython().magic('matplotlib inline')
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
import math
from keras.layers import Conv2D, Flatten
from scipy import signal
tf.python.control_flow_ops = tf
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D, Convolution2D, Input, Lambda, SpatialDropout2D 
from keras import initializations
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.layers import Input, ELU
from pathlib import Path
import json

threshold = 1
col = 64
row = 64
batch_size = 64
EPOCH=7


#Method to read image. CV2 reads images in BGR and the simulator provides images in RGB. Therefore convert to 
#RGB domain
def read_img(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

#Image brigtness changing method, based on Vivek Yadav's [2] approach for changing image brightness
def brightness_images(img):
    post_img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    post_img[:,:,2] = np.multiply(post_img[:,:,2],random_bright)
    post_img = cv2.cvtColor(post_img,cv2.COLOR_HSV2RGB)
    return post_img
# Another approach to adjust brightness used for experimentation 
def brightness_images_2(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
    h, s, v = cv2.split(hsv)
    v += 255
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
# Resize the image to the givin dimensions 
def resize_img(image, col, row):
    image = cv2.resize(image, (col,row), interpolation=cv2.INTER_AREA)
    return image
# Crop away the car hood from the orginal image  
def crop_img(img):
    shape = img.shape
    img = img[0:shape[0]-20,0:shape[1]]
    img = resize_img(img, 64, 64)
    return img
#flip raw and processed images aroung the center , as well as reverse the signal of the steering angel 
def modified_flip(image, steer):
    image=cv2.flip(image,1)
    steer=np.multiply(steer,-1)
    return image, steer

### Loading CSV data

csv_path = 'driving_log.csv'
raw_data = pd.read_csv(csv_path,index_col = False)
raw_data.columns = ['center', 'left', 'right', 'steer', 'throttle', 'brake', 'speed']
raw_steer = np.array(raw_data.steer,dtype=np.float32)
raw_data['steering'] = pd.Series(raw_steer, index=raw_data.index)

#Combine all filters in this method and call it from the keras generator. There are two main methods, one for
#training and the other for validation.

def all_filters_train(generator_csv):
    #Use the left, center and right images randomly from datasets. 
    #The chance of each image to be picked was determined empirically  
    rand_value= np.random.randint(8)
    if (rand_value == 0) or (rand_value == 1) or (rand_value == 2) or (rand_value == 3):
        img_data = generator_csv['center'][0].strip()
        image = cv2.imread(img_data)
        steer_ang = generator_csv['steering'][0] 
    #in 1/4 of cases flip the image and change the steering angle
    if (rand_value == 4):
        img_data = generator_csv['center'][0].strip()
        image = cv2.imread(img_data)
        image = cv2.flip(image,1)
        steer_ang = generator_csv['steering'][0] * -1 
    if (rand_value == 5):
        img_data = generator_csv['left'][0].strip()
        image = cv2.imread(img_data)
        steer_ang = generator_csv['steering'][0] + 0.15
    if (rand_value == 6):
        img_data = generator_csv['right'][0].strip()
        image = cv2.imread(img_data)
        steer_ang = generator_csv['steering'][0] - 0.15 
    if (rand_value == 7):
        img_data = generator_csv['left'][0].strip()
        image = cv2.imread(img_data)
        image = cv2.flip(image,1)
        steer_ang = (generator_csv['steering'][0] * -1) - 0.15 
    if (rand_value == 8):
        img_data = generator_csv['right'][0].strip()
        image = cv2.imread(img_data)
        image = cv2.flip(image,1)
        steer_ang = (generator_csv['steering'][0] * -1) + 0.15 
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = brightness_images(image)
    image = crop_img(image)
    image = np.array(image)
    return image,steer_ang
# Validation filters method combined  
def all_filters_validate(generator_csv):
    img_data = generator_csv['center'][0].strip()
    image = cv2.imread(img_data)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = crop_img(image)
    image = np.array(image)
    return image
# Save model for Keras and generators are based on Vevik Yadav's and Save and Load Your Keras Deep Learning Models
#by Jason Brownlee  

def generate_train_batch(data,batch_size):
    batch_images = np.zeros((batch_size, col, row, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            process_line = np.random.randint(len(data))
            generator_csv= data.iloc[[process_line]].reset_index()
            x,y = all_filters_train(generator_csv)
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering

def generate_validation_patch(data):
    while 1:
        for process_line in range(len(data)):
            generator_csv = data.iloc[[process_line]].reset_index()
            x = all_filters_validate(data)
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            y = generator_csv['steering'][0]
            y = np.array([[y]])
            yield x, y
def save_model(fileModelJSON,fileWeights):
    if Path(fileModelJSON).is_file():
        os.remove(fileModelJSON)
    json_string = model.to_json()
    with open(fileModelJSON,'w' ) as f:
        json.dump(json_string, f)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)
    model.save_weights(fileWeights)


# In[13]:

# deep learning model

def my_final_model():
    model = Sequential()
    input_shape = (col, row, 3)
    model = Sequential()
    #Normalize the images with keras
    model.add(Lambda(lambda x: x/255.-0.5,input_shape=input_shape))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(16, 3, 3, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(24, 3, 3, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(36, 3, 3, border_mode='valid', activation='relu'))
    model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 2, 2, border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(265))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(1, name='output'))
    
    model.summary()
    return model

# Model definition 
model = my_final_model()
#Use adam with 0.0001 learning rate
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam,loss='mse')

#The code is based on Keras's generator implementation, Comma.ai, and Vevik Yadev's 
### threshold reduced over training to include more small angles
valid_generator =generate_validation_patch(raw_data)
val_size = len(raw_data)
threshold = 1
highest_score = 0
best_value = 1000
for i in range(EPOCH):
    train_generator = generate_train_batch(raw_data,batch_size)
    nb_vals = np.round(len(raw_data)/val_size)-1
    #fit_generator(self, generator, samples_per_epoch, nb_epoch, verbose=1, callbacks=None, 
    #validation_data=None, nb_val_samples=None, class_weight=None, max_q_size=10, nb_worker=1, pickle_safe=False, 
    #initial_epoch=0)
    history = model.fit_generator(train_generator,
            samples_per_epoch=50304, nb_epoch=1,validation_data=valid_generator,
                        nb_val_samples=val_size) 
    fileModelJSON = 'model_' + str(i) + '.json'
    fileWeights = 'model_' + str(i) + '.h5'
    
    save_model(fileModelJSON,fileWeights)
    
    loss_value = history.history['val_loss'][0]
    if loss_value < best_value:
        highest_score = i 
        best_value= loss_value
        fileModelJSON = 'model_best.json'
        fileWeights = 'model_best.h5'
        save_model(fileModelJSON,fileWeights)
    
    threshold = 1/(i+1)
print('Best model found at iteration # ' + str(highest_score))
print('Best Validation score : ' + str(np.round(best_value,4)))


# References and Acknowledgment
# 
# Would like to acknowledge 
# [1]http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
# [2]https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.dcwx90st3
# [3]https://carnd-forums.udacity.com/questions/26214464/behavioral-cloning-cheatsheet
# [4]https://arxiv.org/pdf/1608.01230v1.pdf
# [5]https://github.com/commaai/research/blob/master/train_steering_model.py
# [6]https://medium.com/@KunfengChen/training-and-validation-loss-mystery-in-behavioral-cloning-for-cnn-from-udacity-sdc-project-3-dfe3eda596ba#.2mnauogtg
