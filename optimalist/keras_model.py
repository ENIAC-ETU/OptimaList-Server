import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from google.cloud import storage
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from argparse import ArgumentParser
from keras import backend as K

np.random.seed(32)


def get_data(paths, split=True):
    x = []
    y = []

    for path in paths:
        with open(path, 'r') as file:
            data = file.readlines()

        for line in data:
            line = line.split(',')
            if len(line) == 4:
                x.append(np.array(line[:3]))
                y.append(line[3])

    x = np.array(x)
    y = np.array(y)
    if split:
        x, x_validation = np.split(x, [int(len(x) / 5) * 4])
        y, y_validation = np.split(y, [int(len(y) / 5) * 4])
        return x, x_validation, y, y_validation
    else:
        return x, y


def preprocess_data(path):
    with open(path, 'r') as file:
        data = file.readlines()

    with open('train.txt', 'w') as file:
        for line in data:
            line = line.split(',')
            for idx, n in enumerate(line[1:]):
                n = n.strip('\n')
                file.write(n)
                if idx % 4 == 3:
                    file.write('\n')
                else:
                    file.write(',')


def train_mlp(x_train, y_train, x_validation, y_validation, epochs):
    K.clear_session()
    print("Starting training...")
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    early_stop_callback = EarlyStopping(monitor='val_loss',
                  min_delta=0,
                  patience=3,
                  verbose=0, mode='auto')
    callback_list = [
        early_stop_callback,
    ]
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_validation, y_validation),
                        callbacks=callback_list)

    print("Trained successfully.")
    model_json = history.model.to_json()
    with open('keras-model/model.json', 'w') as file:
        file.write(model_json)
    model.save_weights('keras-model/model.h5')
    print("Writed to the file successfully.")

    return history


def train_cnn(x_train, y_train, x_validation, y_validation, epochs):
    K.clear_session()
    print("Starting training...")
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_validation = x_validation.reshape((x_validation.shape[0], x_validation.shape[1], 1))
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # fit model
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0,
                                        patience=5,
                                        verbose=0, mode='auto')
    callback_list = [
        early_stop_callback,
    ]
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_validation, y_validation),
                        callbacks=callback_list)

    print("Trained successfully.")
    model_json = history.model.to_json()
    with open('keras-model/model.json', 'w') as file:
        file.write(model_json)
    model.save_weights('keras-model/model.h5')
    print("Writed to the file successfully.")

    return history


def predict(model, x_input):
    x_input = np.array(x_input)
    x_input = x_input.reshape((1, 3, 1))
    prediction = model.predict(x_input, verbose=0)
    return prediction


def evaluate(x, y, model):
    a = []
    for i in x:
        a.append(model.predict(i.reshape((1, 3)), verbose=0))
    a = np.array(a)
    a = a.reshape((a.shape[0], 1))
    b = y.reshape((y.shape[0], 1))
    c = np.concatenate((a, b), axis=1)

    return c


def upload_model():
    client = storage.Client.from_service_account_json('keras-model/credentials.json')
    bucket = client.get_bucket('optimalist-keras-model')
    model_files = [
        'model.json',
        'model.h5'
    ]
    for model_file in model_files:
        blob = bucket.blob(model_file)
        blob.upload_from_filename('keras-model/{0}'.format(model_file))
    print("Uploaded model files successfully.")


def download_model():
    K.clear_session()
    client = storage.Client.from_service_account_json('keras-model/credentials.json')
    bucket = client.get_bucket('optimalist-keras-model')
    model_files = [
        'model.json',
        'model.h5'
    ]
    for model_file in model_files:
        blob = bucket.blob(model_file)
        blob.download_to_filename('keras-model/{0}'.format(model_file))
    print("Downloaded model files successfully.")


def load_model():
    K.clear_session()
    json_file = open('keras-model/model.json', 'r')
    model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("keras-model/model.h5")
    #model.compile(optimizer='adam', loss='mse')
    print("Loaded model successfully.")
    return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--interval", dest="interval")
    args = parser.parse_args()
    print(args)
    train_paths = [
        'keras-model/data/train-initial-interval-{}.txt'.format(args.interval),
        'keras-model/data/train-products-interval-{}.txt'.format(args.interval)
    ]
    x_train, x_validation, y_train, y_validation = get_data(train_paths)
    history = train_cnn(x_train, y_train, x_validation, y_validation, 500)
    upload_model()
