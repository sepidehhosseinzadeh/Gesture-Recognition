# Hand-Gesture-Recognition

## Task
Developing a classification model to differentiate gestures.

## Data
List of 2000 examples of 6 different types of hand gestures. 
The data files are in “GestureM ExampleN.txt” format, where M is the gesture number (1-6) and N is the example number (1-2000).
Each CSV file contains a 50x8 matrix (50 time-domain samples, 8 channels of data). 
The value of the sample represents the amplitude of the muscle activation.

## Pre-processing the Data
The labels (gesture number 1-6) are extracted from the name of the file using Python’s re library, and converted to categorical numbers, for example if we have 3 classes, labels will be 100, 010, 001.
In order to speed up the process, I pre-processed the data and saved them in a directory “data”.

## Method
Due to the fact that data is time-domain sequences, the best model for sequential data is Long short-term memory (LSTM) deep neural networks. LSTM is a recurrent neural network (RNN) architecture that remembers values over arbitrary intervals.
My first model (lstm rawData.py) is training a LSTM with the raw input data. In recurrent network, the set of context units provides the system with memory in the form of a trace of processing at the previous time slice. 
The pattern of activation on hidden units corresponds to an encoding or internal representation of the input pattern. 
In this way, the context layer holds the history of the input data. The recurrent network uses the history to enable recognition of the time-series data [1]. Following [1], my basic network scheme is shown below:
1. Input layer: number of features (in our case 8 dimensions) nodes 2. Hidden layer: 150 nodes
3. Output layer: number of classes (in our case 6) nodes

## Encoding the Data: Feature Extraction
### Time Domain Features
I also tried to encode the raw data in such a way that time steps remain the same for LSTM network. 
The encoding features are as below. Where N generally is the length of the signal, but I assumed it to be length of the dimensions which is 8, xn is nth dimension.
(1) Mean Absolute Value, (2) Root Mean Square, (3) Variance, (4) Standard Deviation, (5) Waveform Length, (6) Zero Crossing, (7) Slope Sign Change, (8) F_total (summation of all x), 
(9) Willison Amplitude, (10) Mean Absolute Value Slope, (11) Integrated EMG, (12) Wavelet Transform (Computed the value by Python Signal library and got the
sum over second and third dimensions. The reason is to keep the time steps), (13) Skewness.

## Experiments and Results
Train and test split was 25% for test data and the rest for tanning data.
My first model was LSTM with raw input data. It has 80% average train accuracy and 74% test accuracy. Adding 100 hidden layer after 150 layer made the network deeper, and the recognition test accuracy came to 83% average train accuracy and 78% test accuracy. 
Extracting features in along the dimension (to keep the time domain patterns) decreased the test accuracy to 17%. Extracting features along time steps, had the 57% test accuracy.
I should mention that I vary the learning rate, batch size, activation function and optimizer to select the best model.
I tried SVM to classify the features, but it was too slow. I also tried MLP to classify the features.

## Conclusion
The best model for multi-dimensional time domain signals are LSTM neural networks. Hand gesture classification using features, doesn’t have a high accuracy because extracting features remove the time-domain patterns.

## Codes
model1: LSTM rawData.py: 78% average test accuracy (83% average train ac- curacy)
model2: LSTM features.py: 57% average test accuracy
model3: SVM features.py
model4: MLP features.py

## Dependencies
numpy, re, pandas, keras, sklearn, and conda.

## Input
The best model is LSTM rawData.py. Change the “inputDir” variable to be the address the input data. The data in inputData will be split to 25% test and 75% train data.

## Reference
[1] Murakami, Kouichi, and Hitomi Taguchi. “Gesture Recognition Using Recurrent Neu- ral Networks.” In Proceedings of the SIGCHI conference on Human factors in computing systems, pp. 237-242. ACM, 1991.

[2] Rechy-Ramirez, Ericka Janet, and Huosheng Hu. “Stages for Developing Control Sys- tems using EMG and EEG signals: A survey.” School of Computer Science and Electronic Engineering, University of Essex (2011): 1744-8050.

[3] Ververidis, Dimitrios, Sotirios Karavarsamis, Spiros Nikolopoulos, and Ioannis Kompat- siaris. “Pottery gestures style comparison by exploiting Myo sensor and forearm anatomy.” In Proceedings of the 3rd International Symposium on Movement and Computing, p. 3. ACM, 2016.

[4] Hamedi, M., Sh-H. Salleh, A. M. Noor, T. T. Swee, and I. K. Afizam. “Comparison of different time-domain feature extraction methods on facial gestures’ EMGs.” In Prog. Electromagn. Res. Symp. Proc, vol. 12, pp. 1897-1900. 2012.

[5] Ahsan, Md Rezwanul, Muhammad Ibn Ibrahimy, and Othman O. Khalifa. “Electromyg- raphy (EMG) signal based hand gesture recognition using artificial neural network (ANN).” In Mechatronics (ICOM), 2011 4th International Conference On, pp. 1-6. IEEE, 2011.

[6] A. Phinyomark, C. Limsakul, and P. Phukpattaranont. “A novel feature extraction for robust EMG pattern recog- nition. ” Journal of Computing, 1(1):7180, 2009.

[7] H.P. Huang and C.Y. Chen. “Development of a myoelectric discrimination system for a multi-degree prosthetic hand. ” In Robotics and Automation, 1999. Proceedings. 1999 IEEE International Conference on, volume 3, pages 23922397. IEEE, 1999.

[8] M. Weeks. “Digital signal processing using MATLAB and wavelets. ” Jones and Bartlett Publishers, LLC, second edition, 2011.

[9] J.D. Bronzino. The biomedical engineering handbook. CRC Pr I Llc, 2000.
