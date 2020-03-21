from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from scipy import signal
from os import walk
import pandas as pd
import numpy as np
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
    print "Pre processing..."
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

def get_features(X_sample):
    print "Extracting Features..."
    N_samples, N, d = X_sample.shape
    N_features = 8
    features = np.zeros((N_samples, N, N_features), dtype=float)
    for i in range(0, N_samples):
        X = X_sample[i]
        # Mean Absolute Value
        MAV = np.sum(np.abs(X), axis=1) / d
        # Root Mean Square
        RMS = np.sqrt(np.sum(np.power(X,2), axis=1) / d)
        
        # Variance of EMG
        VAR = np.sum(np.power(X, 2), axis=1) / (d-1)
        
        # Standard Deviation
        mu = np.sum(X) / d
        SD = np.sqrt(np.sum(np.power(np.subtract(X, mu), 2), axis=1) / (d-1))
        
        # Waveform Length
        wl = 0.0 # length=d
        for t in range(0, N-1):
            wl += np.abs(np.subtract(X[t],X[t+1]))
        WL = []
        for i in range(0, d):
            WL.append(wl[i])
        WL.append(1)
        WL.append(1)
        WL = WL * 5

        # Zero Crossing
        ZC = len(np.where(np.diff(np.sign(X)))[0])

        # Slope Sign Change
        asign = np.sign(X)
        SSC = len(((np.roll(asign, 1) - asign) != 0).astype(int))

        ### model 2
        # F total
        F_tot = 0.0
        for t in range(0, N):
            F_tot += np.sum(X[t]) / N

        # Willison Amplitude
        WA = 0.0
        mu = np.sum(X, axis=0) / N
        for t in range(0, N-1):
            if np.subtract(np.abs(np.subtract(X[t],X[t+1])), mu).all() >= 0:
               WA += 1

        # Integrated EMG
        IEMG = MAV*N

        # Mean Absolute Value Slope
        MAVS = 0.0
        for t in range(0, N-1):
            MAV_t = np.sum(np.abs(X[t])) / d
            MAV_t1 = np.sum(np.abs(X[t+1])) / d
            MAVS += MAV_t1 - MAV_t

        # Wavelet Transform
        WT = []
        widths = np.arange(1, d+1)
        for t in range(0, N):
            WT.append(signal.cwt(X[t], signal.ricker, widths))
        WT = np.asarray(WT)

        # Skewness
        mu = np.sum(X) / N
        SK = (np.sum(np.power(np.subtract(X, mu), 3)) / N) / (np.power(np.sum(np.power(np.subtract(X, mu), 2) / N), 3.0/2.0))

        f_num = [SSC, F_tot, WA, MAVS, SK] * 10

        f = [MAV, RMS, VAR, SD, WL, IEMG, np.sum(np.sum(WT, axis=1),axis=1), f_num]
        features[i] = np.matrix.transpose(np.asarray(f))

    return features


def build_model():
    # Set the parameters by cross-validation
    parameters = [{'kernel': ['rbf'],
              'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
              'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    print("# Tuning hyper-parameters")
    clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
    return clf


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    X, Y = load_data()
    X, y = shuffle(X, Y, random_state=0)
    X_features = get_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X_features, Y, test_size=0.30)
    
    clf = build_model()

    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on training set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

if __name__ == '__main__':
    main()
