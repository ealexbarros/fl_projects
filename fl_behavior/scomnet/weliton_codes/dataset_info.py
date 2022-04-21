import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import random

def load_data(filename):
    df = pd.read_csv(filename)
    x, y = df.iloc[:, :-1], df.iloc[:, [-1]].values.ravel()
    return df

df = load_data("2018_challenge_cleaned.csv")
df['person'].value_counts().plot()
#plt.show()

df2 = df.dropna()
counts = df['person'].value_counts()
print(len(df.person.value_counts()))

df2 = df[~df['person'].isin(counts[counts < 120].index)]
print(len(df2.person.value_counts()))
df2.to_csv('less_2018_challenge.csv', index=False)

with open('less_2018_challenge.csv', 'r') as file:
    reader = csv.reader(file)
    n_linhas = 0
    header = next(reader)
    row_dict = {}
    if header != None:
        for row in reader:
            #if n_linhas < 10:
            #    print("ROW: ", row)
            #    print("ROW: ", row[-1])
            if row[-1] not in row_dict:
                row_dict[row[-1]] = []
                row_dict[row[-1]].append(row)
            else:
                row_dict[row[-1]].append(row)
            n_linhas = n_linhas + 1

#print("row_dict.keys(): ", row_dict.keys())
count = 0

dataset_100 = []
dataset_200 = []
dataset_500 = []

for chave in row_dict.keys():
    print("chave: ", chave)
    print("count: ", count)
    for linha in row_dict[chave]:
        if count < 100:
            dataset_100.append(linha)
        if count < 200:
            dataset_200.append(linha)
        if count < 500:
            dataset_500.append(linha)
    count = count + 1
    if count > 500:
        break

with open('100_2018_challenge.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(dataset_100)

with open('200_2018_challenge.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(dataset_200)

with open('500_2018_challenge.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(dataset_500)

