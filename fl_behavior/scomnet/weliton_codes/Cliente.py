import flwr as fl
import tensorflow as tf
import pandas as pd
import sys
import csv
import numpy

data_loss_training = []
data_loss_test = []
data_acc_training = []
data_acc_teste = []

def load_data(filename):
    df = pd.read_csv(filename)
    x, y = df.iloc[:, :-1], df.iloc[:, [-1]].values.ravel()
    return x, y

def convertStrToListFloat(stringList):
    final_list_float_numpy = []
    for lista in stringList:
        lista_replace = lista.replace("[","").replace("]","")
        lista_float = [float(s) for s in lista_replace.split(',')]
        final_list_float_numpy.append(lista_float)
    return numpy.asarray(final_list_float_numpy)

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        #Carregar o modelo da memoria
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 20))
        model.set_weights(parameters)
        #model.fit(x_train, y_train, epochs=1, batch_size=26, steps_per_epoch=1)
        dados_resultados = model.fit(x_train, y_train, epochs=1, callbacks=[lr_scheduler])
        print("dados_resultados: ", dados_resultados.history['loss'])
        print("dados_resultados: ", dados_resultados.history['accuracy'])
        data_loss_training.append(dados_resultados.history['loss'])
        data_acc_training.append(dados_resultados.history['accuracy'])
        # SALVAR o modelo da memoria
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        # CARREGAR o modelo da memoria
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        print("LOSS: ", loss)
        print("ACCURACY: ", accuracy)
        data_loss_test.append([loss])
        data_acc_teste.append([accuracy])
        # SALVAR o modelo da memoria
        return loss, len(x_test), {"accuracy": accuracy}


if __name__ == "__main__":
    # Load and compile Keras model
    #model = tf.keras.applications.MobileNetV2((32, 32, 1971), classes=1971, weights=None)  # Cada coluna, 32, Quantidade de Classes, Quantidade de Classes
    #model = tf.keras.applications.MobileNetV2((32, 32, 1971), classes=1971, weights=None)  # Cada coluna, 32, Quantidade de Classes, Quantidade de Classes

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(26,)),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(1971, activation="softmax")
    ])

    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    arquivo = sys.argv[1]
    teste_arquivo = arquivo.split("_")
    teste_arquivo = str(int(teste_arquivo[0])+19)+"_"+teste_arquivo[1]

    # Load dataset
    x_train, y_train = load_data(arquivo)
    y_train = convertStrToListFloat(y_train)

    x_test, y_test = load_data(teste_arquivo)
    y_test = convertStrToListFloat(y_test)

    # Start Flower client
    fl.client.start_numpy_client(server_address="localhost:8080", client=CifarClient())

    #data_loss_training = []
    #data_loss_test = []
    #data_acc_training = []
    #data_acc_teste = []

    with open(arquivo.replace("_dataset.csv","") + '_resultado_loss_training.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_loss_training)

    with open(arquivo.replace("_dataset.csv","") + '_resultado_loss_teste.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_loss_test)

    with open(arquivo.replace("_dataset.csv","") + '_resultado_acc_training.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_acc_training)

    with open(arquivo.replace("_dataset.csv","") + '_resultado_acc_teste.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_acc_teste)



