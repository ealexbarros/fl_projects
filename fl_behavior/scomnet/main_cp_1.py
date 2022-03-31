
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import glob
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import tensorflow_federated as tff

import nest_asyncio
nest_asyncio.apply()

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


# In[26]:


def unique(list1):       
    list_set = set(list1) 
    unique_list = (list(list_set)) 
    unique_list.sort()
    return unique_list

def create_userids( df ):
    array = df.values
    y = array[:, -1]
    return unique( y )


# In[27]:


def split_dataframe(df):
    RANDOM_STATE = 11235
    
    userids = create_userids(df)
    nbclasses = len(userids)    
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures -1 
    X = array[:,0:nfeatures]
    y = array[:,-1]
    
    enc = OneHotEncoder()
    enc.fit(y.reshape(-1,1))
    y = enc.transform(y.reshape(-1, 1)).toarray()
    X = X.reshape(-1, 128, 3)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_STATE)
    
    mini_batch_size = int(min(X_train.shape[0]/10, 32))
        
    X_train = np.asarray(X_train).astype(np.float32)
    X_val = np.asarray(X_val).astype(np.float32)
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    print(mini_batch_size)
    
    BATCH_SIZE = mini_batch_size
    SHUFFLE_BUFFER_SIZE = 100
    
    train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)
    
    return train_ds, val_ds, X_test, y_test, nbclasses    


# In[28]:


def load_data():
    # config
    BASE_PATH = "/home/joaoneto/biometria/csv_files"
    MIN_NUM_ROWS = 50000
    MAX_NUM_ROWS = 100000
    NUM_FRAMES = 250

    users_statistics = pd.read_csv("users_statistics.csv")
    valid_users = users_statistics[(users_statistics.nrows >= MIN_NUM_ROWS) & (users_statistics.nrows <= MAX_NUM_ROWS)]["player_id"].unique()

    tmp_data = []
    for user in valid_users:
        for csv_file_path in glob.glob(f"{BASE_PATH}/{user}/*.csv"):
            tmp_data.append(pd.read_csv(csv_file_path))
            
    data = pd.concat(tmp_data)
    data.reset_index(inplace=True, drop=True)

    users = data['player_id'].unique()
    
    train_set, user_list = split_data(data, users, NUM_FRAMES)
    train_set = np.array([np.array(x) for x in train_set]) 
    train_set_join = train_set.reshape(train_set.shape[0], 384)
    
    data_join = pd.DataFrame(train_set_join)
    data_join['user'] = user_list
    
    train_ds, val_ds, X_test, y_test, n = split_dataframe(data_join)
    
    return train_ds, val_ds, X_test, y_test, n
    
def split_data(data, users, num_frames):
    user_list = []
    train = []
    frame_size = 128
    step = 50

    for user in users:
        data_user = data[data['player_id']==user]  
        data_user = data_user.iloc[:,[0,1,2]]
        
        for w in random.sample(range(0, data_user.shape[0] - frame_size, step), num_frames):
            end = w + frame_size        
            frame = data_user.iloc[w:end,[0, 1, 2]]        
            train.append(frame)
            user_list.append(user)

    return train, user_list


# In[29]:


def get_datasets():
    train_dataset, validation_dataset, X_test, y_test, n = load_data()
    return train_dataset, validation_dataset, X_test, y_test, n


# In[30]:


def centralized_training_loop(train_dataset, validation_dataset, X_test, y_test, nbclasses, input_shape = (128, 3), num_filters = 128):
    
    input_layer = keras.layers.Input(input_shape) 
    
    conv1 = keras.layers.Conv1D(filters=num_filters, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)
    
    conv2 = keras.layers.Conv1D(filters=2*num_filters, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    
    conv3 = keras.layers.Conv1D(num_filters, kernel_size=3,padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    
    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
    output_layer = keras.layers.Dense(nbclasses, activation='softmax')(gap_layer)
    
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    learning_rate = 0.0001
    cb = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=learning_rate)
    
    precision = tf.keras.metrics.Precision(name='precision')
    recall = tf.keras.metrics.Recall(name='recall')
    model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), metrics=['categorical_accuracy', precision, recall]) 
    
    model.summary()
    
    EPOCHS = 500
    
    hist = model.fit(train_dataset, 
                  epochs=EPOCHS,
                  verbose=False, 
                  validation_data=validation_dataset, 
                  callbacks=cb)
    
    hist_df = pd.DataFrame(hist.history)
   
    
    validation_metrics = model.evaluate(validation_dataset, return_dict=True)
    print("Evaluating validation metrics")
    for m in model.metrics:
        print(f"\t{m.name}: {validation_metrics[m.name]:.4f}")        
        
    # EVALUATION 
    X_test = np.asarray(X_test).astype(np.float32)    
    y_true = np.argmax( y_test, axis=1)
    y_pred = np.argmax( model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_true, y_pred)     
    print(accuracy)
    
# In[31]:


def centralized_pipeline():
    train_dataset, validation_dataset, X_test, y_test, n = get_datasets()
    centralized_training_loop(train_dataset, validation_dataset, X_test, y_test, n)


# In[32]:


centralized_pipeline()

