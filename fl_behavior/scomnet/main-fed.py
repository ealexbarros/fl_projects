
# coding: utf-8

# In[2]:

import collections
import functools
import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from typing import Callable, Dict, List, Tuple

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
from tensorflow.keras.utils import to_categorical


# In[3]:


def unique(list1):       
    list_set = set(list1) 
    unique_list = (list(list_set)) 
    unique_list.sort()
    return unique_list

def create_userids( df ):
    array = df.values
    y = array[:, -1]
    return unique( y )


# In[4]:


def split_dataframe(df):
    train_X = df.iloc[:, :384].values
    le = LabelEncoder()
    df['user'] = le.fit_transform(df['user'])
    train_y = to_categorical(df['user']).astype(int)    
    return train_X, train_y


# In[6]:


def create_uniform_dataset(
    X: np.ndarray, y: np.ndarray, number_of_clients: int
) -> Tuple[Dict, tff.simulation.datasets.ClientData]:
    """Function distributes the data equally such that each client holds equal amounts of each class.
    Args:
        X (np.ndarray): Input.\n
        y (np.ndarray): Output.\n
        number_of_clients (int): Number of clients.\n
    Returns:
        [Dict, tff.simulation.ClientData]: A dictionary and a tensorflow federated dataset containing the distributed dataset.
    """
    clients_data = {f"client_{i}": [[], []] for i in range(1, number_of_clients + 1)}
    for i in range(len(X)):
        clients_data[f"client_{(i%number_of_clients)+1}"][0].append(X[i])
        clients_data[f"client_{(i%number_of_clients)+1}"][1].append(y[i])

    return clients_data, create_tff_dataset(clients_data)


# In[7]:


def create_tff_dataset(clients_data: Dict) -> tff.simulation.datasets.ClientData:
    """Function converts dictionary to tensorflow federated dataset.
    Args:
        clients_data (Dict): Inputs.
    Returns:
        tff.simulation.ClientData: Returns federated data distribution.
    """
    client_dataset = collections.OrderedDict()

    for client in clients_data:
        data = collections.OrderedDict(
            (
                ("label", np.array(clients_data[client][1], dtype=np.int32)),
                ("datapoints", np.array(clients_data[client][0], dtype=np.float32)),
            )
        )
        client_dataset[client] = data

    return tff.simulation.FromTensorSlicesClientData(client_dataset)


# In[8]:


def preprocess_dataset(
    epochs: int, batch_size: int, shuffle_buffer_size: int
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
    """Function returns a function for preprocessing of a dataset.
    Args:
        epochs (int): How many times to repeat a batch.\n
        batch_size (int): Batch size.\n
        shuffle_buffer_size (int): Buffer size for shuffling the dataset.\n
    Returns:
        Callable[[tf.data.Dataset], tf.data.Dataset]: A callable for preprocessing a dataset object.
    """

    def _reshape(element: collections.OrderedDict) -> tf.Tensor:

        return (tf.expand_dims(element["datapoints"], axis=-1), element["label"])

    @tff.tf_computation(
        tff.SequenceType(
            collections.OrderedDict(
                label=tff.TensorType(tf.int32, shape=(30,)),
                datapoints=tff.TensorType(tf.float32, shape=(384,)),
            )
        )
    )
    def preprocess(dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Function returns shuffled dataset
        """
        return (
            dataset.shuffle(shuffle_buffer_size)
            .repeat(epochs)
            .batch(batch_size, drop_remainder=False)
            .map(_reshape, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        )

    return preprocess


# In[27]:


def load_data():
    screens = ['Focus', 'Mathisis', 'Memoria', 'Reacton', 'Speedy']
    screens_code = ['1', '2', '3', '4', '5']
    number_of_clients = 10

    base_path = "C:/Users/SouthSystem/Federated Learning/DataBioCom/data"
    phone_accel_file_paths = []

    for directories, subdirectories, files in os.walk(base_path):
        for filename in files:
            if "accel" in filename:
                phone_accel_file_paths.append(f"{base_path}/accel/{filename}")

    data = pd.concat(map(pd.read_csv, phone_accel_file_paths))
    users = data['player_id'].unique()
    
    train_set, user_list = split_data(data, users)
    train_set = np.array([np.array(x) for x in train_set]) 
    train_set_join = train_set.reshape(train_set.shape[0], 384)
    data_join = pd.DataFrame(train_set_join)
    data_join['user'] = user_list
    
    train_X, train_y = split_dataframe(data_join)
    train_client_data, train_data = create_uniform_dataset(train_X, train_y, number_of_clients)
    
    return train_data, len(train_X)

def split_data(data, users):
    user_list = []
    train = []
    frame_size = 128
    step = 50

    for user in users:
        data_user = data[data['player_id']==user]  
        data_user = data_user.iloc[:,[0,1,2]]
        for w in range(0, data_user.shape[0] - frame_size, step):
            end = w + frame_size        
            frame = data_user.iloc[w:end,[0, 1, 2]]        
            train.append(frame)
            user_list.append(user)

    return train, user_list


# In[28]:


def create_keras_model_seq(input_shape, num_filters = 128, nbclasses = 30):
    model = Sequential()
    model.add(keras.layers.Conv1D(filters=num_filters, kernel_size=8, padding='same', activation='relu', input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv1D(filters=2*num_filters, kernel_size=5, padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv1D(num_filters, kernel_size=3,padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(nbclasses, activation='softmax'))
    
    return model


# In[29]:



def iterative_process_fn(
    tff_model: tff.learning.Model,
    server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
    client_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer] = None
) -> tff.templates.IterativeProcess:
    """Function builds an iterative process that performs federated aggregation. The function offers federated averaging, federated stochastic gradient descent and robust federated aggregation.
    Args:
        tff_model (tff.learning.Model): Federated model object.\n
        server_optimizer_fn (Callable[[], tf.keras.optimizers.Optimizer]): Server optimizer function.\n
        aggregation_method (str, optional): Aggregation method. Defaults to "fedavg".\n
        client_optimizer_fn (Callable[[], tf.keras.optimizers.Optimizer], optional): Client optimizer function. Defaults to None.\n
        iterations (int, optional): [description]. Defaults to None.\n
        client_weighting (tff.learning.ClientWeighting, optional): Client weighting. Defaults to None.\n
        v (float, optional): L2 threshold. Defaults to None.\n
        compression (bool, optional): If the model should be compressed. Defaults to False.\n
        model_update_aggregation_factory (Callable[ [], tff.aggregators.UnweightedAggregationFactory ], optional): If the model should be trained with DP. Defaults to None.\n
    Returns:
        tff.templates.IterativeProcess: An Iterative Process.
    """

    return tff.learning.build_federated_averaging_process(
                tff_model,
                server_optimizer_fn=server_optimizer_fn,
                client_optimizer_fn=client_optimizer_fn
            )


# In[30]:


def get_datasets():
    
    train_dataset, n = load_data()
    
    train_preprocess = preprocess_dataset(
        epochs=10,
        batch_size=32,
        shuffle_buffer_size=100,
    )
    
    return train_dataset.preprocess(train_preprocess), n


# In[31]:


train_dataset, len_train_X = get_datasets()


# In[33]:


server_optimizer_fn = get_optimizer(server_optimizer_fn, server_optimizer_lr)
client_optimizer_fn = get_optimizer(client_optimizer_fn, client_optimizer_lr)


# In[32]:


input_spec = train_dataset.create_tf_dataset_for_client(
    train_dataset.client_ids[0]
).element_spec

keras_model_fn = create_keras_model_seq((128,3))
get_keras_model = functools.partial(keras_model_fn)

loss_fn = lambda: tf.keras.losses.CategoricalCrossentropy()
metrics_fn = lambda: [tf.keras.metrics.CategoricalAccuracy()]


def model_fn() -> tff.learning.Model:
    """
    Function that takes a keras model and creates an tensorflow federated learning model.
    """
    return tff.learning.from_keras_model(
        keras_model=get_keras_model(),
        input_spec=input_spec,
        loss=loss_fn(),
        metrics=metrics_fn(),
    )

iterative_process = iterative_process_fn(
    model_fn,
    server_optimizer_fn,
    client_optimizer_fn=client_optimizer_fn,
)

