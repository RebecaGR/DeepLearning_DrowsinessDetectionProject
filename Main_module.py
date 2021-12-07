# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 19:56:41 2021

@author: usuario
"""

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, LSTM, TimeDistributed, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from sklearn.utils import shuffle
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pickle as pkl
import cv2
import os

img_size = 150
path = '.\Data_set'

# Load frames from a single video, function supports max number of frames and frame rate to capture
def load_frames(path,category,max_frames=100,frame_rate=30,resize=(img_size,img_size)):
    
    cap = cv2.VideoCapture(path)
    
    frames = []
    labels = []
    count = 0
     
    while True:
        ret, frame = cap.read()
        
        # Detect if a frame is successfully readed
        if not ret:
            break
        
        if count == max_frames:
            break
        
        frame = cv2.resize(frame,resize)/255.0
    
        frames.append(frame)
        labels.append(category)
        
        del frame
        count += 1
        
    # Good practice to close all windows while releasing the video capture function from OpenCV
    cap.release()
    cv2.destroyAllWindows()
    
    return np.array(frames), np.array(labels)


# Load the labels from each video's file name
def load_category(file):
    
    if file.endswith(('0.mp4','0.mov','0.MOV','0.MP4','0.m4v')):      
        label = 'Alert'
        
    elif file.endswith(('5.mp4','5.mov','5.MOV','5.MP4','5.m4v')):
        label = 'Low Vigilant'
    else:
        label = 'Drowsy'
        
    return label


# Construct the data matrix with each video path and corresponding label
def build_matrix(path):
    #Iterates through the directories
    dirs = os.listdir( path )
    paths_list = []
    label_list = []
    for folder in dirs:
        if folder != '.DS_Store':
            # Each folder contains 3 videos from the same person
            sub_path = os.path.join(path,folder)
            for file in os.listdir(sub_path):
                if file != '.DS_Store':
                    # State the path to each video
                    video_path = os.path.join(sub_path,file)
                    # Load the category depending on the name of the video
                    label = load_category(video_path)
                    # Variables to return the paths to the videos and its labels
                    paths_list.append(video_path)
                    label_list.append(label)
    # Traspose the matrix to return (174,2) --> rows: each video; columns: attribute
    data = np.transpose([paths_list,label_list])
    
    return data

# Convert labels to dictionary 
def define_dict(X, col):
    
    #Define dictionary
    dict_labels = X[:,col]
    labelsNames = np.unique(dict_labels)
    labelsDict = dict(zip(labelsNames,range(len(labelsNames))))
    labels = np.array([labelsDict[cl] for cl in dict_labels])
    
    #change column with dictionary values
    X = np.delete(X,col,1)
    
    return X, labels, labelsDict


# Build matrix with paths and labels
data = build_matrix(path)
# Return dictionary and labels for each video
video_paths, labels_pre, labelsDict = define_dict(data,1)
# Reduce one dimension from video_paths. From: (174,1) to (174,)
video_paths = video_paths.squeeze(axis=1)

# We load the frames into memory
x = []
y = []
for i in range(len(video_paths)):
    f, l = load_frames(video_paths[i], labels_pre[i])
    x.append(f)
    y.append(l)
    del f,l    


# Transform the labels category from integer values to one-hot encoding
labels_one_hot = to_categorical(y)

# Shuffle the matrix with the video_paths and its corresponding labels before the neuronal network
frames_shuffle, labels_shuffle = shuffle(x, labels_one_hot)
# Split the data to get test set (10%)
X_temp, X_tes, y_temp, y_tes = train_test_split(frames_shuffle, labels_shuffle, test_size=0.1, random_state=42)
# Split the data into training and validation (70%-20%   the other 10% of the data is the test set)
X_t, X_v, y_t, y_v = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Delete unused variables from memory
del labels_one_hot, x, y, i


# New shaping for the TimeDistibuted (samples, time-step, width, lenght, channels)
X_train = np.array(X_t)
y_train = np.array(y_t)

X_val = np.array(X_v)
y_val = np.array(y_v)

X_test = np.array(X_tes)
y_test = np.array(y_tes)

del X_t, X_v, X_tes, X_temp, y_t, y_v, y_tes, y_temp


batch = 10
num_epochs = 10

model = Sequential([
    TimeDistributed(Conv2D(32,(5,5),activation="relu",input_shape = (10,img_size,img_size,3))),
    TimeDistributed(MaxPooling2D(2,2)),
    TimeDistributed(Conv2D(16,(3,3),activation = "relu")),
    TimeDistributed(MaxPooling2D(2,2)),
    TimeDistributed(Conv2D(8,(3,3),activation = "relu")),
    TimeDistributed(GlobalAveragePooling2D()),
    TimeDistributed(Dense(256, activation='relu')),
    LSTM(128, activation='relu', return_sequences=True),
    Dense(128,activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64,activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(3,activation="softmax")
    ])

model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics = ['acc'])

# Train the model and evaluate with the validation data
model.fit(X_train,y_train,batch_size=batch, epochs=num_epochs, verbose=1,validation_data=(X_val,y_val))    

# Evaluate trained model with test set of data and obtained generalization accuracy
print('\nEvaluating on test data')
score, acc = model.evaluate(X_test,y_test,batch_size=batch)
print('\nScore: ', score)
print("\nValidation Accuracy: ", acc)

# Model architecture
model.summary()
