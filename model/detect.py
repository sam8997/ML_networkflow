import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, isdir, join
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ignore warnings
import warnings  
warnings.filterwarnings("ignore",category=FutureWarning)

import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend
from tensorflow.keras import callbacks

from argparse import ArgumentParser

from nltk import ngrams
from common import read_csv

lstm_window = 15 

def predict(directory, use_modin=True):
    filename=f"{directory}/full_data.csv"
    print(filename)
    if not os.path.isfile(filename): return

    model = keras.models.load_model(args.model)

    tuples, test_X = read_csv(filename, no_label=True, use_modin=use_modin)
    lstm_X = genLSTM(np.array(test_X))

    out = model.predict(np.array(lstm_X))
    pred_Y = (out > 0.5).ravel()
    malicious = np.sum(pred_Y)
    ratio = malicious/len(test_X)

    print(f'Malicious/Total: {malicious}/{len(test_X)}, ratio: {ratio}')
    tuples = tuples[lstm_window-1:]
    mal_tuples = tuples[pred_Y == 1]
    print(f'Mal packets: {mal_tuples["pkts"].sum()}')
    mal_tuples.to_csv(args.output, index=False)
    backend.clear_session()

    with open('summary.txt', 'a') as f:
        f.write(f'{ratio*100:.3f}%, {malicious}, {filename}\n')

def genLSTM(data, label=None):
    # read training and testing data
    data = list(ngrams(data, lstm_window))
    if label is not None: 
        label = list(ngrams(label[lstm_window-1:], 1))
        return np.array(data), np.array(label)
    else: return np.array(data)

def trainByLSTM(train_data, train_label):
    lstm_train_data, lstm_train_label = genLSTM(train_data, train_label)
    
    # Initialising the RNN
    model = Sequential()

    # Adding the LSTM and Dense layer
    model.add(Bidirectional(LSTM(lstm_train_data.shape[2], return_sequences=False, dropout=0.3), input_shape=(lstm_train_data.shape[1], lstm_train_data.shape[2])))

    model.add(Dense(units = 1, activation='sigmoid'))
    
    # Compiling
    model.compile(optimizer = keras.optimizers.Adam(), loss = 'binary_crossentropy', metrics=['accuracy'])

    # Weight
    norm_sz = len(np.where(train_label == 0.0)[0])
    mal_sz = len(np.where(train_label == 1.0)[0])
    weight_0 = (1/norm_sz) * ((norm_sz+mal_sz) / 2.0)
    weight_1 = (1/mal_sz) * ((norm_sz+mal_sz) / 2.0)
    class_weight = {0: weight_0, 1: weight_1}
    print("Class weight: ", class_weight)

    # Training
    model.fit(lstm_train_data, lstm_train_label, epochs = 5, batch_size = 1024, class_weight=class_weight)

    # Save model
    model.save('lstm.h5')
    print(model.summary())

    backend.clear_session()
     
def testByLSTM(test_data, test_label):
    model = keras.models.load_model(args.model)
    lstm_test_data, lstm_test_label = genLSTM(test_data, test_label)

    predicted_label = model.predict(lstm_test_data)
    # Draw ROC Curve
    fpr, tpr, thresholds = roc_curve(lstm_test_label, predicted_label)
    auc_rf = auc(fpr, tpr)
    print("AUC", auc_rf)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC: {:.3f}'.format(auc_rf))
    plt.savefig('ROC.png')


    predicted_label[predicted_label > 0.5] = 1
    predicted_label[predicted_label <= 0.5] = 0
    
    print('====Confusion_matrix====')
    print('========================')
    print('======Measurement=======')
    tn,fp,fn,tp = confusion_matrix(lstm_test_label, predicted_label).ravel()
    print(f"TN: {tn}, FP: {fp}")
    print(f"FN: {fn}, TP: {tp}")
    print("Accuracy: %f" %(accuracy_score(lstm_test_label, predicted_label)))
    print("FPR: %f" %(fp/(fp+tn)))
    print("TPR: %f" %(tp/(tp+fn)))
    print("Recall: %f" %(recall_score(lstm_test_label, predicted_label)))
    print('========================')
    backend.clear_session()
    return predicted_label

def train(filename):
    # read training and testing data
    train_X, train_Y = read_csv(filename, use_modin=False)
    train_X, train_Y = train_X.to_numpy(), train_Y.to_numpy()

    # Train LSTM
    trainByLSTM(train_X, train_Y)
    # Get LSTM result
    #predicted_label = testByLSTM(train_X, train_Y).reshape(-1)

def test(filename):
    test_X, test_Y = read_csv(filename, use_modin=False)
    predicted_label = testByLSTM(test_X.to_numpy(), test_Y.to_numpy()).reshape(-1)

    test_X = test_X[lstm_window-1:]
    test_Y = test_Y[lstm_window-1:]
    FP = test_X[(predicted_label == 1) & (test_Y == 0)]
    TP = test_X[(predicted_label == 1) & (test_Y == 1)]
    FN = test_X[(predicted_label == 0) & (test_Y == 1)]
    TN = test_X[(predicted_label == 0) & (test_Y == 0)]

    # Get mis-classified data
    FP.to_csv('mal-test.csv')

tune_rate = 5e-5 #Zeus
#tune_rate = 1e-6

def tune(filename):
    tune_X, tune_Y = read_csv(filename, use_modin=False)
    tune_X, tuen_Y = tune_X.to_numpy(), tune_Y.to_numpy()
    tune_data, tune_label = genLSTM(tune_X, tune_Y)
    model = keras.models.load_model(args.model)
    model.compile(optimizer=keras.optimizers.Adam(tune_rate), loss='binary_crossentropy', metrics=['accuracy'])

    norm_sz = len(np.where(tune_label == 0.0)[0])
    mal_sz = len(np.where(tune_label == 1.0)[0])
    weight_0 = (1/norm_sz) * ((norm_sz+mal_sz) / 2.0)
    weight_1 = (1/mal_sz) * ((norm_sz+mal_sz) / 2.0)
    class_weight = {0: weight_0, 1: weight_1}
    print("Class weight: ", class_weight)

    # Training
    model.fit(tune_data, tune_label, epochs=int(args.epoch), batch_size=1024, class_weight=class_weight)
    model.save(f'tune-{args.model}')
    args.model = f'tune-{args.model}'
    test(filename)

def tune_norm(filename):
    tuples, data = read_csv(filename, no_label=True, use_modin=False)

    tune_data, tune_label = genLSTM(np.array(data), np.zeros(data.shape[0]))

    model = keras.models.load_model(args.model)
    model.compile(optimizer=keras.optimizers.Adam(tune_rate), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(tune_data, tune_label, epochs=int(args.epoch), batch_size=4096)
    model.save(f'tune_norm-{args.model}')

def tune_mal(filename):
    tuples, data = read_csv(filename, no_label=True, use_modin=False)

    tune_data, tune_label = genLSTM(np.array(data), label=np.ones(data.shape[0]))

    model = keras.models.load_model(args.model)
    model.compile(optimizer=keras.optimizers.Adam(tune_rate), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(tune_data, tune_label, epochs=int(args.epoch), batch_size=1024)
    model.save(f'tune_mal-{args.model}')


def main():
    train_csv_file = 'train.csv' 

    if args.action == 'train':
        train(train_csv_file)
    elif args.action == 'test':
        test(args.test_file)
    elif args.action == 'predict':
        predict(args.predict_dir, use_modin=False)
    elif args.action == 'tune_norm':
        tune_norm(args.tune_file)
    elif args.action == 'tune_mal':
        tune_mal(args.tune_file)
    elif args.action == 'tune':
        tune(args.tune_file)
    
    
if __name__ == '__main__':
    parser = ArgumentParser(description = 'Detection argument setting.')
    parser.add_argument('-a', '--action', choices=['train', 'predict', 'test', 'tune_norm', 'tune_mal', 'tune'], help='choose actions', dest='action', default='train') 
    parser.add_argument('-t', '--test_file', dest='test_file', default='test.csv')
    parser.add_argument('-p', '--predict_dir', dest='predict_dir', help='Directory of the input to predict')
    parser.add_argument('-m', '--model', dest='model', default='lstm.h5', help='Model to use')
    parser.add_argument('-e', '--epoch', dest='epoch', default='20', help='Tune epochs')
    parser.add_argument('-f', '--tune_file', dest='tune_file', help='File to tune')
    parser.add_argument('-o', '--output', dest='output', default='mal.csv', help='Name of output malicious record')
    args = parser.parse_args()

    main()
