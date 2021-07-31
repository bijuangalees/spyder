# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 05:35:35 2021

@author: bijuangalees
"""

import os
import random
import itertools
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import VGG16, VGG19, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input as process_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as process_vgg19
from tensorflow.keras.applications.resnet50 import preprocess_input as process_resnet

import cv2

from sklearn.metrics import classification_report, confusion_matrix
#################################################33333333


import os
import random
import itertools
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import VGG16, VGG19, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input as process_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as process_vgg19
from tensorflow.keras.applications.resnet50 import preprocess_input as process_resnet

import cv2

from sklearn.metrics import classification_report, confusion_matrix

########################################################################


### DEFINE SOME PARAMETERS ###

base_path = "../input/split-garbage-dataset/split-garbage-dataset/"
labels = os.listdir(base_path + "train/")
SHAPE = (224,224,3)
##################################################################
### PLOT SAMPLE IMAGES ###

f, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize = (16,7))

for ax, label in zip(axes.ravel(), labels):
    img = np.random.choice(os.listdir(base_path + "train/" + label))
    img = cv2.imread(base_path + "train/" + label + "/" + img)
    img = cv2.resize(img, SHAPE[:2])
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGRA2RGB))
    ax.set_title(label)
#####################################################################
### CREATE EMPTY GENERATORS ###

train_datagen = ImageDataGenerator(
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode='nearest'
)

validation_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()\
#######################################################################
### FLOW GENERATORS ###

train_batch_size = 32
val_batch_size = 32

train_generator = train_datagen.flow_from_directory(
            base_path + '/train',
            target_size = (SHAPE[0], SHAPE[1]),
            batch_size = train_batch_size,
            class_mode = 'categorical',
            shuffle = True
)

validation_generator = validation_datagen.flow_from_directory(
            base_path + '/valid',
            target_size = (SHAPE[0], SHAPE[1]),
            batch_size = val_batch_size,
            class_mode = 'categorical',
            shuffle = False
)

test_generator = test_datagen.flow_from_directory(
            base_path + '/test',
            target_size = (SHAPE[0], SHAPE[1]),
            batch_size = val_batch_size,
            class_mode = 'categorical',
            shuffle = False
)

#########################################################################

### RETRIVE TEST LABEL FROM GENERATOR ###

test_num = test_generator.samples

label_test = []
for i in range((test_num // test_generator.batch_size)+1):
    X,y = test_generator.next()
    label_test.append(y)
        
label_test = np.argmax(np.vstack(label_test), axis=1)

label_test.shape

################################################################
def set_seed(seed):
    
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

def import_base_model(SHAPE):

    vgg16 = VGG16(weights = 'imagenet', include_top = False, input_shape = SHAPE)
    vgg19 = VGG19(weights = 'imagenet', include_top = False, input_shape = SHAPE)
    resnet = ResNet50(weights = 'imagenet', include_top = False, input_shape = SHAPE)

    for layer in vgg16.layers[:-4]:
        layer.trainable = False

    for layer in vgg19.layers[:-5]:
        layer.trainable = False

    for layer in resnet.layers[:-10]:
        layer.trainable = False
        
    return vgg16, vgg19, resnet
##############################################
#CONCATENATION
########################################
### GET PRETRAINED MODELS ###


vgg16, vgg19, resnet = import_base_model(SHAPE)
########################
set_seed(33)

inp = Input((224,224,3))

vgg_16_process = Lambda(process_vgg16)(inp)
vgg_16 = vgg16(vgg_16_process)
x_vgg_16 = GlobalMaxPool2D()(vgg_16)
x_vgg_16 = Dense(128, activation='relu')(x_vgg_16)

resnet_process = Lambda(process_resnet)(inp)
res_net = resnet(resnet_process)
x_resnet = GlobalMaxPool2D()(res_net)
x_resnet = Dense(128, activation='relu')(x_resnet)

vgg_19_process = Lambda(process_vgg19)(inp)
vgg_19 = vgg19(vgg_19_process)
x_vgg_19 = GlobalMaxPool2D()(vgg_19)
x_vgg_19 = Dense(128, activation='relu')(x_vgg_19)

x = Concatenate()([x_vgg_16, x_resnet, x_vgg_19])
out = Dense(6, activation='softmax')(x)

model = Model(inp, out)
model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=1e-4), metrics='accuracy')

es = EarlyStopping(monitor='val_accuracy', mode='auto', restore_best_weights=True, verbose=1, patience=7)
model.fit(train_generator, steps_per_epoch = train_generator.samples/train_generator.batch_size,
          epochs=50, validation_data=validation_generator, validation_steps = validation_generator.samples/validation_generator.batch_size, 
          callbacks=[es], verbose = 1)
#############################################################################################

pred_test = np.argmax(model.predict(test_generator), axis=1)

###################################################################################

print(classification_report(label_test, pred_test, target_names=labels))
################################################################
cnf_matrix = confusion_matrix(label_test, pred_test)

plt.figure(figsize=(7,7))
plot_confusion_matrix(cnf_matrix, classes=labels)
plt.show()
############################################################
#SIMPLE AVERAGE
#################################
### GET PRETRAINED MODELS ###

vgg16, vgg19, resnet = import_base_model(SHAPE)
#############
set_seed(33)

inp = Input((224,224,3))

vgg_16_process = Lambda(process_vgg16)(inp)
vgg_16 = vgg16(vgg_16_process)
x_vgg_16 = GlobalMaxPool2D()(vgg_16)
x_vgg_16 = Dense(128, activation='relu')(x_vgg_16)

resnet_process = Lambda(process_resnet)(inp)
res_net = resnet(resnet_process)
x_resnet = GlobalMaxPool2D()(res_net)
x_resnet = Dense(128, activation='relu')(x_resnet)

vgg_19_process = Lambda(process_vgg19)(inp)
vgg_19 = vgg19(vgg_19_process)
x_vgg_19 = GlobalMaxPool2D()(vgg_19)
x_vgg_19 = Dense(128, activation='relu')(x_vgg_19)

x = Average()([x_vgg_16, x_resnet, x_vgg_19])
out = Dense(6, activation='softmax')(x)

model = Model(inp, out)
model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=1e-4), metrics='accuracy')

es = EarlyStopping(monitor='val_accuracy', mode='auto', restore_best_weights=True, verbose=1, patience=7)
model.fit(train_generator, steps_per_epoch = train_generator.samples/train_generator.batch_size,
          epochs=50, validation_data=validation_generator, validation_steps = validation_generator.samples/validation_generator.batch_size, 
          callbacks=[es], verbose = 1)
#######################################################################################
pred_test = np.argmax(model.predict(test_generator), axis=1)

########################################################################################
cnf_matrix = confusion_matrix(label_test, pred_test)

plt.figure(figsize=(7,7))
plot_confusion_matrix(cnf_matrix, classes=labels)
plt.show()
###############################################################

#WEIGHTED AVERAGE
######################################################

### GET PRETRAINED MODELS ###

vgg16, vgg19, resnet = import_base_model(SHAPE)
#########################################################
class LinearW(Layer):

    def __init__(self):
        super(LinearW, self).__init__()
        
    def build(self, input_shape):
        
        self.W = self.add_weight(
                    shape=(1,1,len(input_shape)),
                    initializer='uniform',
                    dtype=tf.float32,
                    trainable=True)
        
    def call(self, inputs):

        # inputs is a list of tensor of shape [(n_batch, n_feat), ..., (n_batch, n_feat)]
        # expand last dim of each input passed [(n_batch, n_feat, 1), ..., (n_batch, n_feat, 1)]
        inputs = [tf.expand_dims(i, -1) for i in inputs]
        inputs = Concatenate(axis=-1)(inputs) # (n_batch, n_feat, n_inputs)
        weights = tf.nn.softmax(self.W, axis=-1) # (1,1,n_inputs)
        # weights sum up to one on last dim

        return tf.reduce_sum(weights*inputs, axis=-1) # (n_batch, n_feat)
    
    
    
######################
set_seed(33)

inp = Input((224,224,3))

vgg_16_process = Lambda(process_vgg16)(inp)
vgg_16 = vgg16(vgg_16_process)
x_vgg_16 = GlobalMaxPool2D()(vgg_16)
x_vgg_16 = Dense(128, activation='relu')(x_vgg_16)

resnet_process = Lambda(process_resnet)(inp)
res_net = resnet(resnet_process)
x_resnet = GlobalMaxPool2D()(res_net)
x_resnet = Dense(128, activation='relu')(x_resnet)

vgg_19_process = Lambda(process_vgg19)(inp)
vgg_19 = vgg19(vgg_19_process)
x_vgg_19 = GlobalMaxPool2D()(vgg_19)
x_vgg_19 = Dense(128, activation='relu')(x_vgg_19)

x = LinearW()([x_vgg_16, x_resnet, x_vgg_19])
out = Dense(6, activation='softmax')(x)

model = Model(inp, out)
model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=1e-4), metrics='accuracy')

es = EarlyStopping(monitor='val_accuracy', mode='auto', restore_best_weights=True, verbose=1, patience=7)
model.fit(train_generator, steps_per_epoch = train_generator.samples/train_generator.batch_size,
          epochs=50, validation_data=validation_generator, validation_steps = validation_generator.samples/validation_generator.batch_size, 
          callbacks=[es], verbose = 1)

### RETRIVE LINEAR WEIGHTS ###

tf.nn.softmax(model.get_weights()[-3]).numpy()

#################################################################
pred_test = np.argmax(model.predict(test_generator), axis=1)
#############################################################
print(classification_report(label_test, pred_test, target_names=labels))
###########################################
cnf_matrix = confusion_matrix(label_test, pred_test)

plt.figure(figsize=(7,7))
plot_confusion_matrix(cnf_matrix, classes=labels)
plt.show()
###################################################




































































































