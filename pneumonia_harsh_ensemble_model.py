# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 05:08:52 2021

@author: bijuangalees
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import shutil
from cv2 import *
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.applications.resnet_v2 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from numpy import expand_dims
###########################################################
tf.keras.backend.clear_session()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#############################################################

kernel = np.ones((5,5), np.uint8)
train_path = 'chest_xray/train/'
new_train_path = 'New Images/Train'
for files in os.listdir(train_path):

    new_train_path2 = os.path.join(new_train_path,files)

    for f in os.listdir(train_path+files):

        img_path=os.path.join(train_path+files,f)
        norm=imread(img_path)
        norm=cvtColor(norm,COLOR_BGR2GRAY)
        th1=equalizeHist(norm)
        eroded = cv2.erode(th1, kernel)
        dilate = cv2.dilate(eroded,kernel)
        eroded2 = cv2.erode(dilate,kernel)

        new_train_path3 = os.path.join(new_train_path2,f)

        imwrite(new_train_path3,eroded2)

test_path = 'D:/chest_xray/test/'
new_test_path = 'D:/New Images/Test'

for files in os.listdir(test_path):

    new_test_path2 = os.path.join(new_test_path,files)

    for f in os.listdir(test_path+files):

        img_path=os.path.join(test_path+files,f)

        norm=imread(img_path)
        norm=cvtColor(norm,COLOR_BGR2GRAY)
        th1 = equalizeHist(norm)
        eroded = cv2.erode(th1, kernel)
        dilate = cv2.dilate(eroded, kernel)
        eroded2 = cv2.erode(dilate, kernel)

        new_test_path3 = os.path.join(new_test_path2,f)

        imwrite(new_test_path3, eroded2)
######################################################################

IMG_SIZE = 300

TRAINING_DIR = "D:/chest_xray/train/"
training_datagen = ImageDataGenerator(rescale = 1/255 ,
                                  #     rotation_range=15,
                                  # height_shift_range=0.2,
                                  # width_shift_range=0.2,
                                  # shear_range=0.2,
                                  zoom_range=0.3,
                                  vertical_flip=True,
                                  fill_mode='nearest')
train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                       target_size=(IMG_SIZE,IMG_SIZE) ,class_mode='binary',
                                                       batch_size=64,shuffle=True )


TEST_DIR = "D:/chest_xray/test/"
test_datagen = ImageDataGenerator(rescale = 1/255)
test_generator = test_datagen.flow_from_directory(TEST_DIR,target_size=(IMG_SIZE,IMG_SIZE), class_mode='binary',
                                                  batch_size=64,
                                                  shuffle=False)

#################################################################################
x,y = train_generator.next()
for i in range(0,1):
    image = x[i]
    plt.imshow(image)
    plt.show()
    
############################################################################
feature_extractor = tf.keras.applications.ResNet50V2(include_top=False,weights='imagenet',input_shape=(IMG_SIZE,IMG_SIZE,3))
feature_extractor.trainable = False
############################################################################## 

model = tf.keras.models.Sequential([
    feature_extractor,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.summary()

#####################################################################################

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs={}):
    if(logs['accuracy']>=0.99):
      self.model.stop_training=True

callbacks=myCallback()
METRICS = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')]

checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, save_weights_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=2, mode='max')

# early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=3, mode='min')

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=METRICS )

history = model.fit(train_generator , epochs=5 , callbacks=[callbacks,checkpoint,lr_reduce], validation_data=test_generator)

############################################################################################
model.evaluate(test_generator, batch_size=32)


###############################################################################

model.save('model_with_975_early_stopping.h5')

#################################################################################

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
precision=history.history['precision']
val_precision=history.history['val_precision']
recall=history.history['recall']
val_recall=history.history['val_recall']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation Precision per epoch
#------------------------------------------------
plt.plot(epochs, precision, 'r', "Training Precision")
plt.plot(epochs, val_precision, 'b', "Validation Precision")
plt.title('Training and validation Precision')
plt.figure()

plt.plot(epochs, recall, 'r', "Training Recall")
plt.plot(epochs, val_recall, 'b', "Validation Recall")
plt.title('Training and validation Recall')

###########################################################################################

model =  tf.keras.models.load_model("model_with_975_early_stopping.h5")
model.summary()
##################################################################################3
#Displaying Incorrect Images
fnames = test_generator.filenames
y_pred = model.predict_classes(test_generator).reshape((-1,))
errors = np.where(y_pred != test_generator.classes)[0]
# print(errors)
incorrect_images = []
for i in errors:
    incorrect_images.append(fnames[i])
    

######################################################################################
from PIL import Image

path = "New Images/Test/"
j=1
my_dpi = 200
fig = plt.figure(figsize=(10, 10), dpi=my_dpi)

for i in incorrect_images:
    name = i.split('\\')
    new_path = path + i
    new_path = new_path.replace(os.sep, '/')
    ax1 = fig.add_subplot(5, 5, j)
    ax1.set_title(name[0])
    # ax1.set_xlabel('X label')
    # ax1.set_ylabel('Y label')
    ax1.set_xticks([])
    ax1.set_yticks([])
    pil_img = imread(new_path)
    ax1.imshow(pil_img)
    j+=1
###################################################################################3

#Predicting on Internet Images
kernel = np.ones((5,5), np.uint8)
val_path = 'New Images/Val'
for files in os.listdir(val_path):

    new_val_path2 = os.path.join(val_path,files)

    for f in os.listdir(new_val_path2):

        img_path=os.path.join(new_val_path2,f)
        norm=imread(img_path)
        norm=cvtColor(norm,COLOR_BGR2GRAY)
        th1=equalizeHist(norm)
        eroded = cv2.erode(th1, kernel)
        dilate = cv2.dilate(eroded,kernel)
        eroded2 = cv2.erode(dilate,kernel)
        eroded2 = resize(eroded2,(IMG_SIZE,IMG_SIZE))
        imwrite(img_path,eroded2)
#########################################################################################
IMG_SIZE=300
VAL_DIR = "New Images/Val"
val_datagen = ImageDataGenerator(rescale = 1/255)
val_generator = val_datagen.flow_from_directory(VAL_DIR,target_size=(IMG_SIZE,IMG_SIZE), class_mode='binary',shuffle=False)
#################################################################################################
model.evaluate(val_generator)
##############################################3
from sklearn.metrics import classification_report, confusion_matrix
Y_pred = model.predict_classes(val_generator)
print('Confusion Matrix')
print(confusion_matrix(val_generator.classes, Y_pred))
print('Classification Report')
target_names = ['NORMAL', 'PNEUMONIA']
print(classification_report(val_generator.classes, Y_pred, target_names=target_names))
##############################################################################################

from sklearn.metrics import classification_report, confusion_matrix
Y_pred = model.predict_classes(test_generator)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, Y_pred))
print('Classification Report')
target_names = ['NORMAL', 'PNEUMONIA']
print(classification_report(test_generator.classes, Y_pred, target_names=target_names))

########################################################################################################
model = tf.keras.applications.ResNet50V2(include_top=False,weights='imagenet',input_shape=(300,300,3))

ixs = [2, 5, 7, 10 ,25]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)
img = load_img('New Images/Train/PNEUMONIA/BACTERIA-49691-0001.jpeg', target_size=(300, 300))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)
feature_maps = model.predict(img)
square = 8
for fmap in feature_maps:
	ix = 1
	for _ in range(square):
		for _ in range(square):
			ax = plt.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
			ix += 1
	plt.show()