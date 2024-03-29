import sys
import numpy as np
import pandas as pd
from datetime import datetime
from contextlib import redirect_stdout

sys.path.append(sys.path[0] + '/../eval/')
from evaluate import evaluate

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

sys.path.append(sys.path[0] + '/../common/')
from helper_functions_DDI import load_data, create_indices, encode, output_entities
from neural_network_DDI import build_network, save_model_and_indices, load_model_and_indices

learn = False
predict = True

# Parameters
MAX_LEN = 100
MODEL_NAME = 'LSTM_DDI'
FULL_PARSE = True
EPOCHS = 15

# parse arguments
trainDir = sys.argv[1]
# valDir = 'data/devel/'
testDir = sys.argv[2]

# timestamp
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%Y-%b-%d-%H%M")
logfile = 'logs/DDI_' + timestampStr + '.log'

""" Learn """
if learn:
    # load training data in a suitable form
    trainData = load_data(trainDir, full_parse=FULL_PARSE)

    idx = create_indices(trainData, MAX_LEN)
    X, y = encode(trainData, idx)

    # convert to one-hot encoding
    y = np.array([to_categorical(i, num_classes=5) for i in y])

    # split into training and validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    # X_train = X
    # y_train = y
    # valData = load_data(valDir, full_parse=FULL_PARSE)
    # X_test, y_test = encode(valData, idx)
    # y_test = np.array([to_categorical(i, num_classes=5) for i in y_test])

    # build network
    model = build_network(idx, full_parse=FULL_PARSE)

    with open(logfile, 'w') as f:
        with redirect_stdout(f):
            model.summary()
        print('\n',file=f)

    # train model
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(
        X_test, y_test), validation_steps=1, verbose=2)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(logfile, mode='a', encoding='utf-8')
    
    # save model and indices, for later use in prediction
    save_model_and_indices(model, idx, MODEL_NAME)


""" Predict """
if predict:
    # load model and associated encoding data
    model, idx = load_model_and_indices(MODEL_NAME)

    # load validation data in a suitable form
    valData = load_data(testDir, full_parse=FULL_PARSE)
    X_test, y_test = encode(valData, idx)

    # tag sentences in dataset
    y_pred = model.predict(X_test)

    # get most likely tag for each word
    tags = list(idx['labels'].keys())
    y_pred = [tags[np.argmax(y)] for y in y_pred]

    # extract entities and write output file
    outfile = 'DDI_' + timestampStr + '.out' if learn else 'changeName.out'
    output_entities(valData, y_pred, 'logs/' + outfile)

    # evaluate using official evaluator
    evaluate('DDI', testDir, outfile)

