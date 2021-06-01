import os
import ast
import tensorflow as tf
import numpy as np
from tensorflow import keras
from helper_functions import *


from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Input
from keras_contrib.layers import CRF
from tensorflow.keras.utils import to_categorical


def build_network(idx):
    '''
    Task: Create network for the learner .

    Input:  idx: index dictionary with word/labels codes, plus maximum sentence length.

    Output: Returns a compiled Keras neural network with the specified layers
    '''
    # sizes
    n_words = len(idx['words'])
    n_labels = len(idx['labels'])
    max_len = idx['maxlen']

    input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words + 1, output_dim=20,
                      input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
    model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(model)  # variational biLSTM
    model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf = CRF(n_labels)  # CRF layer
    out = crf(model)  # output
    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()
    return model

def learn(traindir, validationdir, modelname):
    '''
    Learns a NN model using traindir as training data, and validationdir
    as validation data. Saves learnt model in a file named modelname.
    '''
    # load train and validation data in a suitable form
    traindata = load_data(traindir)
    valdata = load_data(validationdir)

    # create indexes from training data
    max_len = 100
    idx = create_indexs(traindata, max_len)

    # build network
    model = build_network(idx)

    # encode datasets
    X_train = encode_words(traindata, idx)
    y_train = encode_labels(traindata, idx)
    X_test = encode_words(valdata, idx)
    y_test = encode_labels(valdata, idx)


    print("ytrain shape", np.shape(y_train))
    print("ytest shape", np.shape(y_test))
    print("Xtrain shape", np.shape(X_train))
    print("Xtest shape", np.shape(X_test))

    y_train = np.array([to_categorical(i, num_classes=10) for i in y_train])
    y_test = np.array([to_categorical(i, num_classes=10) for i in y_test])

    print("ytrain shape after", np.shape(y_train))
    print("ytest shape after", np.shape(y_test))


    # train model
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), validation_steps=1, verbose=2)

    # save model and indexs, for later use in prediction
    save_model_and_indexs(model, idx, modelname)

def predict(modelname, datadir, outfile):
    '''
    Loads a NN model from file ’modelname’ and uses it to extract drugs
    in datadir. Saves results to ’outfile’ in the appropriate format.
    '''

    # load model and associated encoding data
    model, idx = load_model_and_indexs(modelname)
    # load data to annotate
    testdata = load_data(datadir)

    # encode dataset
    X = encode_words(testdata, idx)

    # tag sentences in dataset
    Y = model.predict(X)
    # get most likely tag for each word
    Y = [[idx['labels'][np.argmax(y)] for y in s] for s in Y]

    # extract entities and dump them to output file
    output_entities(testdata, Y, outfile)

    # evaluate using official evaluator .
    evaluation(datadir, outfile)


def save_model_and_indexs(model, idx, filename):
    '''
    Task: Save given model and indexs to disk

    Input:  model:      Keras model created by _build_network, and trained .
            idx:        A dictionary produced by create_indexs , containing word and
                        label indexes , as well as the maximum sentence length .
            filename:   filename to be created

    Output: Saves the model into filename .nn and the indexes into filename .idx
    '''

    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save model
    model.save('models/'+filename+'.nn')
    
    # Save indexes
    with open('models/'+filename+'.idx', 'w') as f:
        print(idx, file=f)


def load_model_and_indexs(filename):
    '''
    Task:   Load model and associate indexs from disk

    Input:  filename:   filename to be loaded

    Output: Loads a model from filename .nn , and its indexes from filename . idx
            Returns the loaded model and indexes .
    '''
    model = None
    idx = None
    try:
        model = keras.models.load_model('/models/'+filename+'.nn')
    except:
        print("Model named [",filename, ".nn] does not exist." )

    try:
        #read idx and return as dict
        file = open('models/'+filename+'.idx', "r")
        contents = file.read()
        idx = ast.literal_eval(contents)
        file.close()
    except:
        print("Idx named [", filename, ".idx] does not exist.")

    return model, idx