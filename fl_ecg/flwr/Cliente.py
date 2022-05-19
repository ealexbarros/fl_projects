import flwr as fl
import tensorflow as tf
import pandas as pd
import sys
import numpy


from sklearn import preprocessing
from tensorflow.keras import utils


def load_data(filename):
    df = pd.read_csv(filename)
    x, y = df.iloc[:, :-1], df.iloc[:, [-1]].values.ravel()
    return x, y

def convertStrToListFloat(stringList):
    enc = preprocessing.LabelEncoder()
    enc.fit(stringList)
    encode = enc.transform(stringList)
    final_list_float_numpy = []
    for element in encode:
    	final_list_float_numpy.append(element.astype(float))
    return numpy.asarray(final_list_float_numpy)

# Define Flower client
class ECGClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 20))
        model.set_weights(parameters)
        #model.fit(x_train, y_train, epochs=1, batch_size=26, steps_per_epoch=1)
        dados_resultados = model.fit(x_train, y_train, epochs=10, callbacks=[lr_scheduler])
        print("dados_resultados: ", dados_resultados.history['loss'])
        print("dados_resultados: ", dados_resultados.history['accuracy'])
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        print("LOSS: ", loss)
        print("ACCURACY: ", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}


if __name__ == "__main__":
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(26,)),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(1971, activation="softmax")
    ])

    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    arquivo = sys.argv[1]
    print("arquivo: ", arquivo)
    teste_arquivo = arquivo.split("_")
    teste_arquivo = str(int(teste_arquivo[0])+19)+"_"+teste_arquivo[1] #escolher até 20 pra treino
    print("teste_arquivo: ", teste_arquivo)

    # Load dataset
    x_train, y_train = load_data(arquivo)
    y_train = convertStrToListFloat(y_train)

    x_test, y_test = load_data(teste_arquivo)
    y_test = convertStrToListFloat(y_test)

    # Start Flower client
    fl.client.start_numpy_client(server_address="localhost:8080", client=ECGClient())


