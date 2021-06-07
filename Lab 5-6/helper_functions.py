import os
import sys
import numpy as np

# you can comment out this, it just sets my path to the root folder instead of lab5
# sys.path.append('/Users/betty/Desktop/AHLT/AHLT')

from xml.dom.minidom import parse

sys.path.append(sys.path[0] + '/../common/')
from helper_functions_NER import tokenize, get_tag


def load_data_NER(datadir):
    '''
    Task:   Load XML files in given directory, tokenize each sentence,
            and extract ground truth BIO labels for each token.

    Input:  datadir: A directory containing  XML files.

    Output: A dictionary containing the dataset. Dictionary key is sentence_id,
            and the value is a list of token tuples(word, start, end, ground truth).

    Example:
        >>> load_data('data/Train')
        {’DDI - DrugBank.d370.s0 ’: [(’ as ’, 0, 1, ’O’), (’ differin ’, 3, 10, ’B-brand’),
                                    (’ gel ’, 12, 14, ’O ’), ..., (’ with ’, 343, 346, ’O ’),
                                    (’ caution ’, 348, 354, ’O ’), (’.’, 355, 355, ’O ’)],
        ’DDI - DrugBank.d370.s1 ’: [(’ particular ’, 0, 9, ’O ’), (’ caution ’, 11, 17, ’O ’),
                                    (’ should’, 19, 24, ’O ’), ...,(’ differin ’, 130, 137, ’B-brand’),
                                    (’ gel’, 139, 141, ’O ’), (’.’, 142, 142, ’O ’)],
            ...
        }
    '''

    dataset = dict()
    # process each file in directory
    for f in os.listdir(datadir):
        # parse XML file , obtaining a DOM tree
        tree = parse(datadir + f)
        # process each sentence in the file
        for s in tree.getElementsByTagName("sentence"):
            sid = s.attributes["id"].value  # get sentence id
            stext = s.attributes["text"].value  # get sentence text
            # load ground truth entities
            gold = []
            for e in s.getElementsByTagName("entity"):
                # for discontinuous entities, we only get the first span
                offset = e.attributes["charOffset"].value
                (start, end) = offset.split(";")[0].split("-")
                gold.append((int(start), int(end), e.attributes["type"].value))
            # tokenize text
            tokens = tokenize(stext)
            output = [list(token) for token in tokens]
            # print features in format suitable for the learner/classifier
            for i in range(0, len(tokens)):
                # see if the token is part of an entity, and which part (B/I)
                output[i].append(get_tag(tokens[i], gold))

            dataset[sid] = [tuple(out) for out in output]

    return dataset


def create_indices(dataset, max_length):
    '''
    Task:   Create index dictionaries both for input (words) and output (labels)
            from given dataset.

    Input:  dataset:    dataset produced by load_data
            max_length: maximum length of a sentence (longer sentences will
                        be cut, shorter ones will be padded).

    Output: A dictionary where each key is an index name (e.g. " words ", " labels "),
            and the value is a dictionary mapping each word / label to a number.
            An entry with the value for maxlen is also stored.

    Example:
        >>> create_indices(traindata)
            {’words’: { ’<PAD > ’:0 , ’<UNK >’:1 , ’11-day’:2 , ’murine’:3 , ’criteria’: 4 ,
            ’stroke ’:5 , ... ,’ levodopa’: 8511 , ’terfenadine’: 8512}
            ’labels ’: {’<PAD > ’:0 , ’B- group ’:1 , ’B- drug_n ’:2 , ’I- drug_n ’:3 , ’O ’:4 ,
            ’I- group ’:5 , ’B- drug ’:6 , ’I- drug ’:7 , ’B- brand ’:8 , ’I- brand ’:9}
            ’maxlen ’ : 100
            }
    '''
    words = {
        '<PAD>': 0,
        '<UNK>': 1
    }
    # TODO: Add lemmas, PoS, etc.

    labels = {
        '<PAD>': 0,
        '0': 1,
        'B-group': 2,
        'I-group': 3,
        'B-drug_n': 4,
        'I-drug_n': 5,
        'B-drug': 6,
        'I-drug': 7,
        'B-brand': 8,
        'I-brand': 9
    }

    # iterate over all tokens
    counter = 2
    for _, value in dataset.items():
        for word_tuple in value:
            if not word_tuple[0] in words.keys():
                words[word_tuple[0]] = counter
                counter += 1

    idx = {
        'words': words,
        'labels': labels,
        'maxlen': max_length
    }

    return idx


def encode(dataset, idx):
    words_encoded = np.zeros((len(dataset), 100))
    labels_encoded = np.zeros((len(dataset), 100))

    # iterate sentences
    for i, item in enumerate(dataset.items()):
        for j, word in enumerate(item[1]):
            if word[0] in idx['words'].keys():
                words_encoded[i, j] = idx['words'][word[0]]
                labels_encoded[i, j] = idx['labels'][word[3]]
            else:
                words_encoded[i, j] = 1      # Word unknown
            # shorten long sentences
            if j >= idx['maxlen'] - 1:
                break

    return words_encoded, labels_encoded


def output_entities(dataset, preds, outfile):
    '''
    Task: Output detected entities in the format expected by the evaluator

    Input:  dataset:    A dataset produced by load_data .
            preds:      For each sentence in dataset, a list with the labels for each
                        sentence token, as predicted by the model

    Output: prints the detected entities to stdout in the format required by the evaluator .

    Example:
        >>> output \ _entities (dataset, preds)
            DDI-DrugBank.d283.s4|14-35|bile acid sequestrants|group
            DDI-DrugBank.d283.s4|99-104|tricor|group
            DDI-DrugBank.d283.s5|22-33|cyclosporine|drug
            DDI-DrugBank.d283.s5|196-208|fibrate drugs|group
            ...
    '''
    # define a class to write a line to the file
    class Prediction:
        def __init__(self, sid) -> None:
            self.sid = sid
            self.reset()

        def reset(self) -> None:
            self.start = None
            self.end = None
            self.token = ''
            self.tag = None

        def flush(self) -> None:
            offset = str(self.start) + '-' + str(self.end)
            
            if self.tag not in ['drug', 'drug_n', 'group', 'brand']:
                self.reset()
                return

            with open(outfile, 'a') as f:
                print(self.sid + "|" + offset + "|" + self.token + "|" + self.tag, file=f)

            self.reset()


    # iterate over sentences
    for y_sen, data in zip(preds, dataset.items()):
        # iterate over tokens of sentence
        sid, tokens = data
        pred = Prediction(sid)
        for idx, token in enumerate(tokens):
            # get prediction
            label = y_sen[idx]

            # check if continues previous token
            if not (label == '0' or label == '<PAD>'):
                pos, tag = label.split('-')
                if pos == 'B' or tag != pred.tag:   # end run
                    pred.flush()
                    # add new
                    pred.start = token[1]
                    pred.end = token[2]
                    pred.tag = tag
                    pred.token = token[0]
                elif pos == 'I':    # continue run FIXME: currently never being called
                    pred.end = token[2]
                    pred.token += ' ' + token[0]
                else:
                    raise Exception
            else:
                pred.flush()
        
        else:
            pred.flush()



