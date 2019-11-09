import numpy as np
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def min_max_scaler(x):
    for i in range(x.shape[1]):
        x[:, i, :] = MinMaxScaler().fit_transform(x[:, i, :])
    return x


def load_and_partition(data_path, labels_path, nclasses):
    labels = np.load(labels_path)
    dataset = np.load(data_path)
    encoder = preprocessing.LabelEncoder()
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

    encoder.fit(y_train)
    y_train_c = encoder.transform(y_train)
    y_test_c = encoder.transform(y_test)
    y_train_c = np_utils.to_categorical(y_train_c, nclasses)
    y_test_c = np_utils.to_categorical(y_test_c, nclasses)

    x_train = min_max_scaler(x_train)
    x_test = min_max_scaler(x_test)
    return x_train, x_test, y_train_c, y_test_c, encoder.classes_


def baseline_model(neurons, nclasses):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(321, 324), implementation=2, recurrent_dropout=0))
    model.add(Dense(nclasses))
    model.add(Activation('softmax'))

    optimizer = SGD()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def classify(model, x_train, x_test, y_train, y_test, true_classes):
    epochs = 2000
    batch_size = 1000
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test), verbose=1)
    Y_pred = model.predict(x_test)
    # Assign most probable label
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred, target_names=true_classes))
    print(confusion_matrix(y_true, y_pred))


if __name__ == '__main__':
    labels_path = '../data/labels.npy'
    dataset_path = '../data/padded_sequences.npy'
    nclasses = 5
    x_train, x_test, y_train, y_test, classes = load_and_partition(dataset_path, labels_path, nclasses)
    model = baseline_model(64, nclasses)
    classify(model, x_train, x_test, y_train, y_test, classes)
