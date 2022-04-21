import pandas as pd
import csv
import sys
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import random

def load_data(filename):
    df = pd.read_csv(filename)
    x, y = df.iloc[:, :-1], df.iloc[:, [-1]].values.ravel()
    labelencoder = LabelEncoder()
    classe_encoder = labelencoder.fit_transform(y)
    classe_dummy = np_utils.to_categorical(classe_encoder)
    return x, y, classe_dummy

def split_data(filename, numero_linhas):
    with open(filename, 'r') as file:
        # empty dictionary
        row_dict = {}

        reader = csv.reader(file)
        #header = next(reader)
        n_linhas = 0
        itt = 0
        new_dataset_rows = []
        shuffle_dataset_rows = []
        for row in reader:
            print("ROW: ", row)
            row_dict[row[-1]] = []
            shuffle_dataset_rows.append(row)
            #n_linhas = n_linhas + 1

        random.shuffle(shuffle_dataset_rows)
        for row in shuffle_dataset_rows:
            new_dataset_rows.append(row)
            n_linhas = n_linhas + 1
            if n_linhas >= numero_linhas:
                with open(str(itt) + '_dataset_500.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(new_dataset_rows)
                n_linhas = 0
                new_dataset_rows = []
                itt = itt + 1

    #print("Numero de Classes: ", len(row_dict.keys()))
    #print("Numero de Linhas: ", n_linhas)


#split_data("500_2018_challenge.csv", 1500)

x, y, y_dumm = load_data("100_2018_challenge.csv")
#print("y_dumm: ", y_dumm)

for a, b in zip(y, y_dumm):
    print(a," ",b)