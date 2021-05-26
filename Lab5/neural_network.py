import os
import ast
import tensorflow as tf
from tensorflow import keras
from helper_functions import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def build_network (idx):
    '''
    Task: Create network for the learner .

    Input:  idx: index dictionary with word/labels codes, plus maximum sentence length.

    Output: Returns a compiled Keras neural network with the specified layers
    '''
    # sizes
    n_words = len(idx['words'])
    n_labels = len(idx['labels'])
    max_len = idx['maxlen']


    model_lstm.add(Embedding(input_dim=max_len, output_dim=256, input_length = n_words))
    model_lstm.add(SpatialDropout1D(0.3))
    model_lstm.add(LSTM(256, dropout=0.3, recurrent_dropout=0.3))
    model_lstm.add(Dense(256, activation='relu'))
    model_lstm.add(Dropout(0.3))
    model_lstm.add(Dense(n_labels, activation='softmax'))
    model_lstm.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']
    )

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

    # train model
    model.fit(X_train, y_train , validation_data =(X_test, y_test))

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
    
    #Save model
    model.save('models/'+filename+'.nn')
    
    #Save indexes
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