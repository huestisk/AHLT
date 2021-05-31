import os
import sys
import numpy as np
# you can comment out this, it just sets my path to the root folder instead of lab5
sys.path.append('/Users/betty/Desktop/AHLT/AHLT')

from xml.dom.minidom import parse
import re
import random
from nltk.tokenize import word_tokenize
specialchars = ('(', ')', '[', ']', '*', '+', '?')


def load_data(datadir):
    '''
    Task:   Load XML files in given directory, tokenize each sentence, and extract
            learning examples (tokenized sentence + entity pair).

    Input:  datadir: A directory containing XML files.

    Output: A list of classification cases. Each case is a list containing sentence
            id, entity1 id, entity2 id, ground truth relation label, and a list
            of sentence tokens (each token containing any needed information: word,
            lemma, PoS, offsets, etc).

    Example:
        >>> load_data(’data/Train’)
            [[’DDI-DrugBank.d66.s0’,
            ’DDI-DrugBank.d66.s0.e0’, ’DDI-DrugBank.d66.s0.e1’, ’null’,
            [(’<DRUG1>’, ’<DRUG1>’, ’<DRUG1>’), (’-’, ’-’, ’:’),
            (’Concomitant’, ’concomitant’,’JJ’), (’use’,’use’,’NN’),
            (’of’, ’of’, ’IN’), (’<DRUG2>’, ’<DRUG2>’, ’<DRUG2>’), (’and’, ’and’, ’CC’),
            (’<DRUG_OTHER>’, ’<DRUG_OTHER>’, ’<DRUG_OTHER>’),
            (’may’, ’may’, ’MD’),
            ..., (’syndrome’, ’syndrome’, ’NN’), (’.’, ’.’, ’.’)
            ]]
            ...
            [’DDI-MedLine.d94.s12’,
            ’DDI-MedLine.d94.s12.e1’, ’DDI-MedLine.d94.s12.e2’, ’effect’,
            [(’The’, ’the’, ’DT’), (’uptake’, ’uptake’, ’NN’),
            (’inhibitors’, ’inhibitor ’, ’NNS’),
            (’<DRUG_OTHER>’, ’<DRUG_OTHER>’, ’<DRUG_OTHER>’) , (’and’, ’and’, ’CC’),
            (’<DRUG1 >’,’ <DRUG1 >’,’ <DRUG1 > ’) ,
            ... (’effects’, ’effect’, ’NNS’), (’of’, ’of’, ’IN’),
            (’<DRUG2>’, ’<DRUG2>’, ’<DRUG2>’), (’in’, ’in’, ’IN’), ...
            ]]
            ...
    '''

    # tokens now return offsets, but lemma PoS and possible others not included
    classification_cases = []
    for f in os.listdir(datadir):
        if not f.endswith('.xml'):
            continue

        # parse XML file , obtaining a DOM tree
        tree = parse(datadir + f)
        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:

            # get sentence id
            sid = s.attributes["id"].value
            # we're only considering pairs
            pairs = s.getElementsByTagName("pair")
            if len(pairs) > 0:
                for p in pairs:
                    case = []
                    e1 = p.attributes['e1'].value
                    e2 = p.attributes['e2'].value
                    # if ddi exist, get type
                    if p.attributes["ddi"].value == 'true':
                        ddi = p.attributes["type"].value
                    else:
                        ddi = 'null'
                    text = s.attributes["text"].value  # get sentence text
                    # tokenize text
                    tokens = tokenize(text)
                    #if ddi is not 'null':
                    #    print("sentence:",sid,"pair", e1,"-", e2, ddi)

                    case.append([sid, e1, e2, ddi, tokens])
                    classification_cases.append(case)
                    
    return classification_cases



def encode_words(dataset, idx):
    '''
    Task:   Encode the words in a sentence dataset formed by lists of tokens into
            lists of indexes suitable for NN input.

    Input:  dataset:    A dataset produced by load_data .
            idx:        A dictionary produced by create_indexs, containing word and
                        label indexes , as well as the maximum sentence length .

    Output: The dataset encoded as a list of sentence , each of them is a list of
            word indices. If the word is not in the index, <UNK> code is used. If
            the sentence is shorter than max_len it is padded with <PAD > code .

    Example:
        >>> encode_words(traindata, idx)
            [[6882 1049 4911 ... 0 0 0]
            [2290 7548 8069 ... 0 0 0]
            ...
            [2002 6582 7518 ... 0 0 0]]
    '''

    # Note: You may adapt this function to return more than one list per
    # sentence if you want to use different inputs (lemma, PoS, suffixes...)


def encode_labels(dataset, idx):
    '''
    Task:   Encode the ground truth labels in a dataset of classification examples
            (sentence + entity pair).

    Input:  dataset:    A dataset produced by load_data.
            idx:        A dictionary produced by create_indexs, containing word and
                        label indexes, as well as the maximum sentence length .

    Output: The dataset encoded as a list DDI labels, one per classification example.

    Example:
        >>> encode_labels(traindata, idx)
            [[0] [0] [2] ... [4] [0] [0] [1] [0]]
            [[ [4] [6] [4] [4] [4] [4] ... [0] [0]]
            [[4] [4] [8] [4] [6] [4] ... [0] [0]]
            ...
            [[4] [8] [9] [4] [4] [4] ... [0] [0]]
            ]
    '''

    # Note: The shape of the produced list may need to be adjusted depending
    # on the architecture of your network and the kind of output layer you use.


def create_indexs(dataset, max_length):
    '''
    Task:   Create index dictionaries both for input (words) and output (labels)
            from given dataset.

    Input:  dataset: dataset produced by load_data.
            max_length: maximum length of a sentence (longer sentences will
            be cut, shorter ones will be padded).

    Output: A dictionary where each key is an index name (e.g. "words", "labels"),
            and the value is a dictionary mapping each word/label to a number.
            An entry with the value for maxlen is also stored
    Example:
        >>> create_indexs(traindata)
            {’words’: {’<PAD>’:0, ’<UNK>’:1, ’11-day’:2, ’murine’:3, ’criteria’:4,
            ’stroke’:5,...,’levodopa’:8511, ’terfenadine’:8512}
            ’labels’: {’null’:0, ’mechanism’:1, ’advise’:2, ’effect’:3, ’int’:4}
            ’maxlen’: 100
            }
    '''

    # Add ’<PAD>’:0 and ’<UNK>’:1 codes to ’words’ index. The coding of the
    # rest of the words/labels is arbitrary. You may add to the dictionary entries
    # with indexes for other elements you want to use (lemmas, PoS, etc)
    # This indexes will be needed by the predictor to properly use the model

def tokenize(s):
    """
    Task: Given a sentence, calls nltk.tokenize to split it into tokens, and adds to each token its start/end offsetin the original sentence.
    Input:  s: string containing the text for one sentence

    Output: Returns a list of tuples (word , offsetFrom , offsetTo)

    Example:
        tokenize (" Ascorbic acid , aspirin , and the commoncold .")
            [(" Ascorbic ",0,7), ("acid",9,12), (",",13,13), ("aspirin ",15,21),
            (",",22,22), ("and",24,26), ("the",28,30), (" common ",32,37), ("cold ",39,42),("." ,43 ,43)]
    """
    shift = 0
    tokens = []
    tokens0 = word_tokenize(s)
    # find beginning and end of each word
    for word in tokens0:
        if word.startswith(specialchars):
            word = "\\" + word # FIXME
        ans = re.search(word, s[shift:])
        if ans is None:
            continue
        start = ans.start() + shift
        shift += ans.end()
        tokens.append((word, start, shift-1))
    return tokens