import os
import pickle
import tensorflow as tf

import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
from tensorflow.python.keras.models import Sequential
from tensorflow_addons.text import crf_log_likelihood, crf_decode

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Bidirectional, InputLayer

EMBEDDING_SIZE = 40
HIDDEN_SIZE = 40
NUM_LSTM = 3


class CRF(L.Layer):
    def __init__(self,
                 output_dim,
                 sparse_target=True,
                 **kwargs):
        """    
        Args:
            output_dim (int): the number of labels to tag each temporal input.
            sparse_target (bool): whether the the ground-truth label represented in one-hot.
        Input shape:
            (batch_size, sentence length, output_dim)
        Output shape:
            (batch_size, sentence length, output_dim)
        """
        super(CRF, self).__init__(**kwargs)
        self.output_dim = int(output_dim)
        self.sparse_target = sparse_target
        self.input_spec = L.InputSpec(min_ndim=3)
        self.supports_masking = False
        self.sequence_lengths = None
        self.transitions = None

    def build(self, input_shape):
        assert len(input_shape) == 3
        f_shape = tf.TensorShape(input_shape)
        input_spec = L.InputSpec(min_ndim=3, axes={-1: f_shape[-1]})

        if f_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `CRF` '
                             'should be defined. Found `None`.')
        if f_shape[-1] != self.output_dim:
            raise ValueError('The last dimension of the input shape must be equal to output'
                             ' shape. Use a linear layer if needed.')
        self.input_spec = input_spec
        self.transitions = self.add_weight(name='transitions',
                                           shape=[self.output_dim,
                                                  self.output_dim],
                                           initializer='glorot_uniform',
                                           trainable=True)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    def call(self, inputs, sequence_lengths=None, training=None, **kwargs):
        sequences = tf.convert_to_tensor(inputs, dtype=self.dtype)
        if sequence_lengths is not None:
            assert len(sequence_lengths.shape) == 2
            assert tf.convert_to_tensor(sequence_lengths).dtype == 'int32'
            seq_len_shape = tf.convert_to_tensor(
                sequence_lengths).get_shape().as_list()
            assert seq_len_shape[1] == 1
            self.sequence_lengths = K.flatten(sequence_lengths)
        else:
            self.sequence_lengths = tf.ones(tf.shape(inputs)[0], dtype=tf.int32) * (
                tf.shape(inputs)[1]
            )

        viterbi_sequence, _ = crf_decode(sequences,
                                         self.transitions,
                                         self.sequence_lengths)
        output = K.one_hot(viterbi_sequence, self.output_dim)
        return K.in_train_phase(sequences, output)

    @property
    def loss(self):
        def crf_loss(y_true, y_pred):
            y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)
            log_likelihood, self.transitions = crf_log_likelihood(
                y_pred,
                tf.cast(K.argmax(y_true),
                        dtype=tf.int32) if self.sparse_target else y_true,
                self.sequence_lengths,
                transition_params=self.transitions,
            )
            return tf.reduce_mean(-log_likelihood)
        return crf_loss

    @property
    def accuracy(self):
        def viterbi_accuracy(y_true, y_pred):
            # -1e10 to avoid zero at sum(mask)
            mask = K.cast(
                K.all(K.greater(y_pred, -1e10), axis=2), K.floatx())
            shape = tf.shape(y_pred)
            sequence_lengths = tf.ones(shape[0], dtype=tf.int32) * (shape[1])
            y_pred, _ = crf_decode(y_pred, self.transitions, sequence_lengths)
            if self.sparse_target:
                y_true = K.argmax(y_true, 2)
            y_pred = K.cast(y_pred, 'int32')
            y_true = K.cast(y_true, 'int32')
            corrects = K.cast(K.equal(y_true, y_pred), K.floatx())
            return K.sum(corrects * mask) / K.sum(mask)
        return viterbi_accuracy

    def compute_output_shape(self, input_shape):
        tf.TensorShape(input_shape).assert_has_rank(3)
        return input_shape[:2] + (self.output_dim,)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'sparse_target': self.sparse_target,
            'supports_masking': self.supports_masking,
            'transitions': K.eval(self.transitions)
        }
        base_config = super(CRF, self).get_config()
        return dict(base_config, **config)


def build_network(idx, verbose=True):
    '''
    Task: Create network for the learner.

    Input:  idx: index dictionary with word/labels codes, plus maximum sentence length.

    Output: Returns a compiled Keras neural network with the specified layers
    '''

    n_words = len(idx['words'])
    n_labels = len(idx['labels'])
    max_len = idx['maxlen']

    model = Sequential()
    model.add(InputLayer(input_shape=(max_len,)))
    # 50-dim embedding
    model.add(Embedding(input_dim=n_words, output_dim=EMBEDDING_SIZE,
                        input_length=max_len, mask_zero=True))
    # variational biLSTM
    for _ in range(NUM_LSTM):
        model.add(Bidirectional(LSTM(units=HIDDEN_SIZE,
                                 return_sequences=True, recurrent_dropout=0.1)))
    # dense layer
    model.add(Dense(n_labels, activation="relu"))
    # CRF layer
    crf = CRF(n_labels, sparse_target=True)
    model.add(crf)

    model.compile(optimizer='adam', 
                  loss=crf.loss, 
                  metrics=[crf.accuracy])
    
    if verbose:
        model.summary()

    return model


def save_model_and_indices(model, idx, filename):
    '''
    Task: Save given model and indices to disk

    Input:  model:      Keras model created by _build_network, and trained .
            idx:        A dictionary produced by create_indices , containing word and
                        label indexes , as well as the maximum sentence length .
            filename:   filename to be created

    Output: Saves the model into filename .nn and the indexes into filename .idx
    '''

    if not os.path.exists('models'):
        os.makedirs('models')

    # Save model
    model.save_weights('models/' + filename)

    # Save indexes
    with open('models/' + filename + '.idx', 'wb') as f:
        pickle.dump(idx, file=f)


def load_model_and_indices(filename):
    '''
    Task:   Load model and associate indices from disk

    Input:  filename:   filename to be loaded

    Output: Loads a model from filename .nn , and its indexes from filename . idx
            Returns the loaded model and indexes .
    '''

    # load idx
    with open('models/' + filename + '.idx', 'rb') as f:
        idx = pickle.load(f)

    # create model from loaded weights
    model = build_network(idx, verbose=False)
    model.load_weights('models/' + filename)

    return model, idx
