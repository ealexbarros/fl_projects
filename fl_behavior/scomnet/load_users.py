
# coding: utf-8

# In[2]:


import pandas as pd
import glob

base_path = "/home/joaoneto/biometria/csv_files"

users_statistics = pd.read_csv("users_statistics.csv")
valid_users = users_statistics[(users_statistics.nrows >= 50000) & (users_statistics.nrows <= 400000)]["player_id"].unique()
print(valid_users)
tmp_data = []
for user in valid_users:
    for csv_file_path in glob.glob(f"{base_path}/{user}/*.csv"):
        tmp_data.append(pd.read_csv(csv_file_path))
data = pd.concat(tmp_data)
data.reset_index(inplace=True, drop=True)


# In[5]:




