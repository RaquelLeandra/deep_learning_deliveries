import datetime
import os
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Dense, Activation, Masking
from keras.layers import SimpleRNN, LSTM, GRU
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sns.set()


def experiment_resource(experiment_name):
    results_path = '../results/experiments/'
    os.makedirs(results_path, exist_ok=True)
    return os.path.join(results_path, experiment_name)


def confusion_matrix_resource(experiment_name):
    results_path = '../results/experiments/confusion_matrix/'
    os.makedirs(results_path, exist_ok=True)
    return os.path.join(results_path, experiment_name)


def history_resource(experiment_name):
    results_path = '../results/experiments/history/'
    os.makedirs(results_path, exist_ok=True)
    return os.path.join(results_path, experiment_name)


def model_resource(experiment_name):
    results_path = '../results/models/'
    os.makedirs(results_path, exist_ok=True)
    return os.path.join(results_path, experiment_name)


def write_the_model(model, experiment_name):
    with open(experiment_resource(experiment_name + '.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


def min_max_scaler(x):
    for i in range(x.shape[1]):
        missing_values = x[:, :, i] == -1
        x[:, :, i] = ((x[:, :, i]) - np.min(x[:, :, i])) / (np.max(x[:, :, i]) - np.min(x[:, :, i]))
        for j in range(missing_values.shape[0]):
            for k in range(missing_values.shape[1]):
                if missing_values[j, k]:
                    x[j, k, i] = -1
    return x


def load_and_partition(data_path, labels_path, nclasses, scaler=True):
    labels = np.load(labels_path)
    dataset = np.load(data_path)
    encoder = preprocessing.LabelEncoder()
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

    encoder.fit(y_train)
    y_train_c = encoder.transform(y_train)
    y_test_c = encoder.transform(y_test)
    y_train_c = np_utils.to_categorical(y_train_c, nclasses)
    y_test_c = np_utils.to_categorical(y_test_c, nclasses)
    if scaler:
        x_train = min_max_scaler(x_train)
        x_test = min_max_scaler(x_test)
    return x_train, x_test, y_train_c, y_test_c, encoder.classes_


def baseline_model(neurons, nclasses, input_shape=(321, 324), dropout=0):
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=input_shape))
    model.add(SimpleRNN(neurons, input_shape=input_shape, implementation=2, dropout=dropout))
    model.add(Dense(nclasses))
    model.add(Activation('softmax'))

    optimizer = SGD()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def small_LSTM(neurons, nclasses, dropout=0, input_shape=(321, 324)):
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=input_shape))
    model.add(LSTM(neurons, input_shape=input_shape, implementation=2, dropout=dropout))
    model.add(Dense(nclasses))
    model.add(Activation('softmax'))

    optimizer = SGD()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def small_GRU(neurons, nclasses, dropout=0, input_shape=(321, 324)):
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=input_shape))
    model.add(GRU(neurons, input_shape=input_shape, implementation=2, dropout=dropout))
    model.add(Dense(nclasses))
    model.add(Activation('softmax'))

    optimizer = SGD()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def plot_training_history(history, experiment_name):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(history_resource(experiment_name + '_hist_accuracy'), bbox_inches='tight', dpi=200)
    plt.close()
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(history_resource(experiment_name + '_hist_loss'), bbox_inches='tight', dpi=200)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, experiment_name='experiment_cm'):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(data=cm, cmap=cmap, annot=True, xticklabels=classes, yticklabels=classes, linewidths=.5)
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45)
    plt.savefig(confusion_matrix_resource(experiment_name), bbox_inches='tight', dpi=200)
    plt.close()


def classify(model, x_train, x_test, y_train, y_test, true_classes, experiment_name='experiment', epochs=200,
             batch_size=100):
    write_the_model(model, experiment_name)

    init_time = time()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2, verbose=1)
    plot_training_history(history, experiment_name)
    training_time = time() - init_time

    Y_pred = model.predict(x_test)
    score = model.evaluate(x_test, y_test, verbose=0)
    # Assign most probable label
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred, target_names=true_classes))
    print(confusion_matrix(y_true, y_pred))

    model_json = model.to_json()
    with open(model_resource(experiment_name + '_model.json'), 'w') as json_file:
        json_file.write(model_json)
    weights_file = experiment_name + "_weights_" + str(score[1]) + ".hdf5"
    model.save_weights(model_resource(weights_file), overwrite=True)
    with open(experiment_resource(experiment_name + '.txt'), 'a') as fh:
        fh.write('\nHistory params: {}\n'.format(history.params))
        fh.write('\nTraining time: {}\n'.format(datetime.timedelta(seconds=training_time)))
        fh.write('\ntest loss: {}\n'.format(score[0]))
        fh.write('test accuracy: {}\n'.format(score[1]))
        fh.write('Analysis of results:\n')
        fh.write(classification_report(y_true, y_pred, target_names=true_classes))
        fh.write('\nConfusion matrix:\n')
        fh.write(str(confusion_matrix(y_true, y_pred)))
    plot_confusion_matrix(y_true, y_pred, true_classes, experiment_name=experiment_name)
    return score[0], score[1]


def run_test_experiment(x_train, x_test, y_train, y_test, classes):
    nclasses = len(classes)
    model = baseline_model(64, nclasses)
    classify(model, x_train, x_test, y_train, y_test, classes, experiment_name='try_that_works', epochs=5)


def run_all_experiments(x_train, x_test, y_train, y_test, classes, add_to_name=''):
    models = {'baseline': baseline_model,
              'small_lstm': small_LSTM,
              'small_gru': small_GRU
              }
    number_of_neurons = [32, 64, 128, 256, 512]
    nclasses = len(classes)
    dropouts = [0, 0.2, 0.3, 0.4, 0.5]
    results_accuracy = pd.DataFrame(columns=models.keys(), index=number_of_neurons)
    results_loss = pd.DataFrame(columns=models.keys(), index=number_of_neurons)
    for dropout in dropouts:
        for model_name in models:
            for n_neurons in number_of_neurons:
                model = models[model_name](n_neurons, nclasses, input_shape=(321, 100), dropout=dropout)
                experiment_name = add_to_name + '_{}_{}_{}'.format(model_name, n_neurons,
                                                                   str(dropout).replace('.', '_'))
                print(experiment_name)
                loss, accuracy = classify(model, x_train, x_test, y_train, y_test, classes, experiment_name, epochs=300)
                results_accuracy.loc[n_neurons, model_name] = accuracy
                results_loss.loc[n_neurons, model_name] = loss

        results_accuracy.to_csv(experiment_resource(add_to_name + '_results_accuracy_all_{}.csv'.format(dropout)))
        results_loss.to_csv(experiment_resource(add_to_name + '_results_loss_all_{}.csv'.format(dropout)))


if __name__ == '__main__':
    labels_path = '../data/labels.npy'
    dataset_path = '../data/padded_sequences.npy'
    nclasses = 5
    # x_train, x_test, y_train, y_test, classes = load_and_partition(dataset_path, labels_path, nclasses,
    #                                                                scaler=False)

    pca_dataset_path = '../data/pca_sequences.npy'
    x_train, x_test, y_train, y_test, classes = load_and_partition(pca_dataset_path, labels_path, nclasses,
                                                                   scaler=False)
    classes = [cl.replace('The Hunger Games: Catching Fire', 'Hunger Games') for cl in classes]
    run_all_experiments(x_train, x_test, y_train, y_test, classes, 'pca')
