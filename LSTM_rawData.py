from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers.core import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import shuffle
from scipy import signal
from os import walk
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import pdb
import os

# Initalization
np.random.seed(7)

inputDir = "/home/sepideh/Desktop/Tasks_Interview/thamicLab/gesture_data/"
dataDir = inputDir + "data/"
if not os.path.exists(dataDir):
    os.makedirs(dataDir)

nClasses = 6
nFeatures = 8
nSteps = 50

# Functions
def preprocessing(inputDir):
    print "Pre-processing..."
    pattern = re.compile("Gesture(?P<gesture_type>\d+)_Example(?P<example_number>\d+).txt")
    X = []
    Y = []
    for dir_path, dir_names, file_names in walk(inputDir):
        for file_name in file_names:
            example = pd.read_csv(dir_path+file_name, header=None).as_matrix()
            example_n_time_steps, example_n_dim = example.shape
            if example_n_time_steps != nSteps:
                missing_values = np.zeros((nSteps - example_n_time_steps, nFeatures))
                example = np.vstack((example, missing_values))
            X.append(example)
            gesture_type, example_number = pattern.match(file_name).groups()
            Y.append(int(gesture_type))

    X = np.stack(X)
    Y = to_categorical(np.array(Y))
    np.save(dataDir+"X.npy", X)
    np.save(dataDir+"Y.npy", Y)
    return X, Y

def to_categorical(y):
    n_classes = np.max(y)
    y_cat = np.zeros((len(y), n_classes))
    for i in range(0, len(y)):
        y_cat[i, y[i]-1] = 1.0
    
    return y_cat

def load_data():
    for dir_path, dir_names, file_names in walk(dataDir):
        if "X.npy" not in file_names or "Y.npy" not in file_names:
            return preprocessing(inputDir)
    print "Loading Files..."
    return np.load(dataDir+"X.npy"), np.load(dataDir+"Y.npy")

def build_model(first_layer_neurons, second_layer_neurons):
    model = Sequential()
    model.add(LSTM(first_layer_neurons, input_dim=nFeatures, dropout_U=0.3))
    model.add(Dense(second_layer_neurons))
    model.add(Dropout(0.2))
    model.add(Dense(nClasses, activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model


def predict(model, X_test, y_test = None):
    predictions = model.predict(X_test)
    get_class = lambda classes_probabilities: np.argmax(classes_probabilities) + 1
    y_pred = np.array(map(get_class, predictions))
    if y_test is not None:
       y_true = np.array(map(get_class, y_test))
       print accuracy_score(y_true, y_pred)
    return y_pred

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    X, y = load_data()
    X, y = shuffle(X, y, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    
    model = build_model(150, 100)
    model.fit(X_train, y_train, nb_epoch=100, batch_size=100, verbose=2)
    
    scores = model.evaluate(X_train, y_train)
    print("%s: %.2f" % (model.metrics_names[1], scores[1]))
    
    predict(model, X_test, y_test)

if __name__ == '__main__':
    main()

