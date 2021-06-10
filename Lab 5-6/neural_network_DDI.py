import os
import pickle
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Embedding, Conv1D, GlobalMaxPooling1D, Dense

def build_network(idx, full_parse=False, verbose=True):
    '''
    Task: Create network for the learner.

    Input:  
        idx: index dictionary with word/labels codes, plus maximum sentence length.

    Output: 
        Returns a compiled Keras neural network with the specified layers
    '''

    # sizes
    n_words = len(idx['words'])
    n_labels = len(idx['labels'])
    max_len = idx['maxlen'] if not full_parse else idx['maxlen'] * 3

    embedding_dim = 50
    hidden_size = 128

    model = Sequential()
    model.add(InputLayer(input_shape=(max_len,)))
    model.add(Embedding(input_dim=n_words, output_dim=embedding_dim, input_length=max_len))
    model.add(Conv1D(hidden_size, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(n_labels, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    if verbose:
        model.summary()

    return model


def save_model_and_indices(model, idx, filename):
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
    model.save('models/' + filename)

    # Save indexes
    with open('models/' + filename + '.idx', 'wb') as f:
        pickle.dump(idx, file=f)


def load_model_and_indices(filename):
    '''
    Task:   Load model and associate indexs from disk

    Input:  filename:   filename to be loaded

    Output: Loads a model from filename .nn , and its indexes from filename . idx
            Returns the loaded model and indexes .
    '''
    
    # load idx
    with open('models/' + filename + '.idx', 'rb') as f:
        idx = pickle.load(f)

    # load model
    model = keras.models.load_model('models/' + filename)
    
    return model, idx



