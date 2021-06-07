from helper_functions import *
import numpy as np
import tensorflow as tf

from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense

def build_network(idx):
    '''
    Task:   Create network for the learner.
    Input:  idx:    index dictionary with word/labels codes, plus maximum sentence
                    length .

    Output: Returns a compiled Keras neural network with the specified layers
    '''
    # sizes
    n_words = len(idx['words'])
    n_labels = len(idx['labels'])
    max_len = idx['maxlen']

    # create network layers
    inp = Input(shape=(max_len,))
     ## ... add missing layers here ... #
    out = 0# final output layer

    from keras.models import Sequential
    from keras import layers
    embedding_dim = 100
    vocab_size = n_words + 1

    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(n_labels, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # create and compile model
    #model = Model(inp, out)
    #model.compile() # set appropriate parameters ( optimizer , loss , etc )

    return model

    #DDI is not a sequence tagging task (which assign one label per
    # word), but a sentence classification, where a single label is
    # assigned to the whole sentence (or sentence + entity pair in this case).

    #The problem may be approached with an LSTM, but since it
    # produces a label per word, some layers need to be added to
    # convert the output to a single class. A good alternative is using
    # a CNN, which also produces good results for text processing,
    # and are more straightforward to apply to this kind of tasks

    #You will need to add one Embedding layer after the input, that
    # is where the created indexes will become handy.

def learn(traindir, validationdir, modelname):
    '''
    3 learns a NN model using traindir as training data , and validationdir
    4 as validation data . Saves learnt model in a file named modelname
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
    Xtrain = encode_words(traindata, idx)
    Ytrain = encode_labels(traindata, idx)
    Xval = encode_words(valdata, idx)
    Yval = encode_labels(valdata, idx)
    
    print("xtrain", np.shape(Xtrain))
    print("Ytrain", np.shape(Ytrain))
    print("Xval", np.shape(Xval))
    print("Yval", np.shape(Yval))


    #Yval = tf.convert_to_tensor(Yval, dtype=tf.int64)
    #Ytrain = tf.convert_to_tensor(Ytrain, dtype=tf.int64)

    # train model
    model.fit(Xtrain, Ytrain, validation_data=(Xval, Yval))

    # save model and indexs , for later use in prediction
    save_model_and_indexs(model, idx, modelname)

def predict(modelname, datadir, outfile):
    '''
    Loads a NN model from file ’modelname ’ and uses it to extract drugs
    in datadir . Saves results to ’outfile ’ in the appropriate format .
    '''

    # load model and associated encoding data
    model , idx = load_model_and_indexs(modelname)
    # load data to annotate
    testdata = load_data(datadir)

    # encode dataset
    X = encode_words(testdata, idx)

    # tag sentences in dataset
    Y = model.predict(X)
    # get most likely tag for each pair
    Y = [[idx['labels'][np.argmax(y)] for y in Y]]

    # extract entities and dump them to output file
    output_interactions(testdata, Y, outfile)

    # evaluate using official evaluator .
    evaluation(datadir, outfile)

    # Note: Observe the output structure (one class per sentence+pair),
    # is different from the NER task (one class per token).


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
    model.save('models/' + filename + '.nn')

    # Save indexes
    with open('models/' + filename + '.idx', 'w') as f:
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
        model = keras.models.load_model('/models/' + filename + '.nn')
    except:
        print("Model named [", filename, ".nn] does not exist.")

    try:
        # read idx and return as dict
        file = open('models/' + filename + '.idx', "r")
        contents = file.read()
        idx = ast.literal_eval(contents)
        file.close()
    except:
        print("Idx named [", filename, ".idx] does not exist.")

    return model, idx

import numpy as np

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                                        vector, dtype=np.float32)
                                        [:embedding_dim]

    return embedding_matrix

