
# coding: utf-8

# In[93]:


import glob
import os
import pandas as pd
import numpy as np

def split(data, frame_size=128, step=50):
    train = []
    user_list = []
    for user, group_data in data.groupby("player_id"):
        for w in range(0, group_data.shape[0] - frame_size, step):
            end = w + frame_size        
            frame = group_data.iloc[w:end,[0, 1, 2]]        
            train.append(frame)
            user_list.append(user)
    return train, user_list

base_path = "/home/joaoneto/biometria/csv_files"

data = pd.concat([ pd.read_csv(csv_file_path) for csv_file_path in glob.glob(f"{base_path}/*/*.csv")])
# print(data)

train_set, user_list = split(data)
train_set = np.array([np.array(x) for x in train_set])
train_set_join = train_set.reshape(train_set.shape[0], 384)
data_join = pd.DataFrame(train_set_join)
data_join["user"] = user_list
print(data_join[data_join["user"] == "4fjfjlm"])

train_ds, val_ds, X_test, y_test, n = split_dataframe(data_join)

