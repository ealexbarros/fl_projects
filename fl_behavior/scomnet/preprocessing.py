
# coding: utf-8

# In[1]:


import os, shutil, glob
import ujson as json
import pandas as pd
import csv
import timeit
import numpy as np


# In[2]:


path = '/home/joaoneto/biometria/sensors'
users_path = [ f.path for f in os.scandir(path) if f.is_dir() ]
screens = ['Focus', 'Mathisis', 'Memoria', 'Reacton', 'Speedy']
screens_code = ['1', '2', '3', '4', '5']

list_dir = os.listdir(path)
users_list = []
for sub_dir in list_dir:
    users_list.append(sub_dir)


# In[3]:


def convert_to_csv(signal, path, users):
    users_processed = 0
    for i in range(0, len(users)):
        users_processed += 1
        print('Progress: {}/{} users processed'.format(users_processed, len(users)))

        json_files = [pos_json for pos_json in os.listdir(users_path[i]) if pos_json.endswith('.json')]
        
        data_signal = pd.DataFrame(columns=['x', 'y', 'z', 'screen', 'player_id', 'timestamp'])

        for file in json_files:
            js = file.replace('.json','')
            arr = js.split('_')

            with open(users_path[i] + "/" + file,'r') as f:
                data = json.loads(f.read())

            df = pd.json_normalize(data, record_path =[path],
                meta=['player_id']
            )
            df['timestamp'] = arr[1]          
            
            data_signal = data_signal.append(df, ignore_index=True)
            
        x_signal = f'x_{signal}'
        y_signal = f'y_{signal}'
        z_signal = f'z_{signal}'    

        new = [x_signal, y_signal, z_signal, 'screen', 'player_id', 'timestamp']
        new_df = pd.DataFrame(data_signal.values, data_signal.index, new)
        
        # print(new_df['player_id'].value_counts())
        
        filter_values = new_df['screen'].str.contains('|'.join(screens),regex=True)
        new_df_filter = new_df[filter_values]
        saving_directory = '/home/joaoneto/biometria/{arr[0]}_{signal}.csv'
        new_df_filter.to_csv(saving_directory, index=False)


# In[4]:


convert_to_csv(signal='accel', path='accelerometer', users=users_list)
convert_to_csv(signal='gyro', path='gyroscope', users=users_list)

