# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 23:10:52 2019

@author: 11078
"""

import os
os.chdir("D:/Code/Kaggle/Leaf")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

# 用来处理分类和图像数据
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img

root = 'leaf-classification/'
np.random.seed(2016)
split_random_state = 7
split = .9

#----------------------------------------------------
#----------------------加载数据-----------------------
#----------------------------------------------------

def load_numeric_training(standardize=True):
    """
    Loads the pre-extracted features for the training data
    and returns a tuple of the image ids, the data, and the labels
    """
    # Read data from the CSV file
    data = pd.read_csv(os.path.join(root, 'train.csv'))
    ID = data.pop('id')

    # Since the labels are textual, so we encode them categorically
    y = data.pop('species')
    y = LabelEncoder().fit(y).transform(y)
    # standardize the data by setting the mean to 0 and std to 1
    X = StandardScaler().fit(data).transform(data) if standardize else data.values

    return ID, X, y


def load_numeric_test(standardize=True):
    """
    Loads the pre-extracted features for the test data
    and returns a tuple of the image ids, the data
    """
    test = pd.read_csv(os.path.join(root, 'test.csv'))
    ID = test.pop('id')
    # standardize the data by setting the mean to 0 and std to 1
    test = StandardScaler().fit(test).transform(test) if standardize else test.values
    return ID, test


def resize_img(img, max_dim=96):
    """
    Resize the image to so the maximum side is of size max_dim
    Returns a new image of the right size
    """
    # Get the axis with the larger dimension
    max_ax = max((0, 1), key=lambda i: img.size[i])
    # Scale both axes so the image's largest dimension is max_dim
    scale = max_dim / float(img.size[max_ax])
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))


def load_image_data(ids, y=None, max_dim=96, center=True):
    """
    Takes as input an array of image ids and loads the images as numpy
    arrays with the images resized so the longest side is max-dim length.
    If center is True, then will place the image in the center of
    the output array, otherwise it will be placed at the top-left corner.
    """
    # Initialize the output array
    # NOTE: Theano users comment line below and
    X = np.empty((len(ids), max_dim, max_dim, 1))
    # X = np.empty((len(ids), 1, max_dim, max_dim)) # uncomment this
    for i, idee in enumerate(ids):
        # Turn the image into an array
        x = resize_img(load_img(os.path.join(root, 'images', str(idee) + '.jpg'), grayscale=True), max_dim=max_dim)
        x = img_to_array(x)
        # Get the corners of the bounding box for the image
        # NOTE: Theano users comment the two lines below and
        length = x.shape[0]
        width = x.shape[1]
        # length = x.shape[1] # uncomment this
        # width = x.shape[2] # uncomment this
        if center:
            h1 = int((max_dim - length) / 2)
            h2 = h1 + length
            w1 = int((max_dim - width) / 2)
            w2 = w1 + width
        else:
            h1, w1 = 0, 0
            h2, w2 = (length, width)
        # Insert into image matrix
        # NOTE: Theano users comment line below and
        X[i, h1:h2, w1:w2, 0:1] = x
    # Scale the array values so they are between 0 and 1
    return np.around(X / 255.0, 4), y

import tensorflow as tf
def data_augmentation(X_imgs, y):
    X_scale_data = central_scale_images(X_imgs, scales = [0.9])
    y_scale_data = y
    X_translated_arr = translate_images(X_imgs)
    y_translated_arr = np.concatenate((y,y,y,y), axis = 0)
    X_rotate = rotate_images(X_imgs)
    y_rotate = np.concatenate((y,y,y), axis = 0)
    
    X_data = np.concatenate((X_scale_data,X_translated_arr,X_rotate), axis=0)
    y_data = np.concatenate((y_scale_data,y_translated_arr,y_rotate), axis=0)
    return X_data, y_data

# 以中心放缩图片   
def central_scale_images(X_imgs, scales):
    # Various settings needed for Tensorflow operation
    IMAGE_SIZE=96
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([IMAGE_SIZE, IMAGE_SIZE], dtype = np.int32)
    
    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, IMAGE_SIZE, IMAGE_SIZE, 1))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)
    
    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    return X_scale_data
    
def get_translate_parameters(index):
    IMAGE_SIZE=96
    if index == 0: # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype = np.float32)
        size = np.array([IMAGE_SIZE, np.ceil(0.8 * IMAGE_SIZE)], dtype = np.int32)
        w_start = 0
        w_end = int(np.ceil(0.8 * IMAGE_SIZE))
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 1: # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype = np.float32)
        size = np.array([IMAGE_SIZE, np.ceil(0.8 * IMAGE_SIZE)], dtype = np.int32)
        w_start = int(np.floor((1 - 0.8) * IMAGE_SIZE))
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 2: # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype = np.float32)
        size = np.array([np.ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype = np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = int(np.ceil(0.8 * IMAGE_SIZE)) 
    else: # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype = np.float32)
        size = np.array([np.ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype = np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = int(np.floor((1 - 0.8) * IMAGE_SIZE))
        h_end = IMAGE_SIZE 
        
    return offset, size, w_start, w_end, h_start, h_end

## 移动图片
def translate_images(X_imgs):
    offsets = np.zeros((len(X_imgs), 2), dtype = np.float32)
    n_translations = 4
    X_translated_arr = []
    IMAGE_SIZE=96
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), IMAGE_SIZE, IMAGE_SIZE, 1), dtype = np.float32)
            X_translated.fill(1.0) # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i)
            offsets[:, :] = base_offset 
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)
            
            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0], w_start: w_start + size[1], :] = glimpses
            X_translated_arr.extend(X_translated)
    X_translated_arr = np.array(X_translated_arr, dtype = np.float32)
    return X_translated_arr

# 旋转图片
def rotate_images(X_imgs):
    IMAGE_SIZE = 96
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 1))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
                X_rotate.append(rotated_img)
        
    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate


def load_train_data(split=split, random_state=None):
    """
    Loads the pre-extracted feature and image training data and
    splits them into training and cross-validation.
    Returns one tuple for the training data and one for the validation
    data. Each tuple is in the order pre-extracted features, images,
    and labels.
    """
    # Load the pre-extracted features
    ID, X_num_tr, y = load_numeric_training()
    X_num_tr=np.concatenate((X_num_tr,X_num_tr,X_num_tr,X_num_tr,X_num_tr,X_num_tr,X_num_tr,X_num_tr), axis=0)
    # Load the image data
    X_img_tr, y = load_image_data(ID, y)
    
    # 数据增强
    X_img_tr, y = data_augmentation(X_img_tr, y)
    # Split them into validation and cross-validation
    #sss = StratifiedShuffleSplit(n_splits=1, train_size=split, random_state=random_state)
    #train_ind, test_ind = next(sss.split(X_num_tr, y))
    #X_num_val, X_img_val, y_val = X_num_tr[test_ind], X_img_tr[test_ind], y[test_ind]
    #X_num_tr, X_img_tr, y_tr = X_num_tr[train_ind], X_img_tr[train_ind], y[train_ind]
    return X_num_tr, X_img_tr, y



def load_test_data():
    """
    Loads the pre-extracted feature and image test data.
    Returns a tuple in the order ids, pre-extracted features,
    and images.
    """
    # Load the pre-extracted features
    ID, X_num_te = load_numeric_test()
    # Load the image data
    X_img_te, y_none = load_image_data(ID)
    return ID, X_num_te, X_img_te

print('Loading the training data...')
(X_num_tr, X_img_tr, y_tr) = load_train_data(random_state=split_random_state)
y_tr_cat = to_categorical(y_tr)
print('Training data loaded!')

ID, X_num_te, X_img_te = load_test_data()

#----------------------------------------------------
#------------------整合CNN和特征数据-------------------
#----------------------------------------------------

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Input
from keras.layers.merge import concatenate
from keras.layers import Conv2D

def combined_model():

    # Define the image input
    image = Input(shape=(96, 96, 1), name='image')
    # Pass it through the first convolutional layer
    # x = Convolution2D(8, 5, 5, input_shape=(96, 96, 1), border_mode='same')(image)
    x = Conv2D(8, (5, 5), input_shape=(96, 96, 1), padding='same')(image)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    # Now through the second convolutional layer
    x = (Conv2D(32, (5, 5), padding='same'))(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    # Flatten our array
    x = Flatten()(x)
    # Define the pre-extracted feature input
    numerical = Input(shape=(192,), name='numerical')
    # Concatenate the output of our convnet with our pre-extracted feature input
    concatenated = concatenate([x, numerical])

    # Add a fully connected layer just like in a normal MLP
    x = Dense(100, activation='relu')(concatenated)
    x = Dropout(.5)(x)

    # Get the final output
    out = Dense(99, activation='softmax')(x)
    # How we create models with the Functional API
    model = Model(inputs=[image, numerical], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

print('Creating the model...')
model = combined_model()
print('Model created!')


#----------------------------------------------------
#--------------------训练模型-------------------------
#----------------------------------------------------

# 测试集和训练集
from sklearn.model_selection import train_test_split
X_num_train, X_num_test, X_img_train, X_img_test, y_train, y_test = train_test_split(X_num_tr, X_img_tr, y_tr, test_size=0.1)

# 训练模型 保存最好的
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
best_model_file = "leaf-classification/leafnet.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True, period=1)
batch_size, epochs = 128, 20
history = model.fit([X_img_tr, X_num_tr], y_tr_cat,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          shuffle=True,
          callbacks=[best_model])
print(history.history.keys())
history.history['val_loss']
history.history['loss']
history.history['val_acc']
history.history['acc']


# 加载训练好的模型，在测试集上用一下
from keras.models import save_model(load_mode)
model = load_model('leaf-classification/leafnet.h5')
loss,acc = model.evaluate([X_img_tr, X_num_tr], y_tr_cat,batch_size=batch_size,verbose=0)
print("loss:{}, acc:{}".format(loss, acc))
y_pre_aug = model.predict([X_img_te, X_num_te])

submit = pd.read_csv('leaf-classification/sample_submission.csv')
submit.iloc[:,1:] = y_pre_aug
submit.to_csv('leaf-classification/submission_by_cnn_aug.csv', index=False)
