import os
import sys
import numpy as np
# you can comment out this, it just sets my path to the root folder instead of lab5
sys.path.append('/Users/betty/Desktop/AHLT/AHLT')

from xml.dom.minidom import parse
from common.helper_functions_NER import *

def load_data(datadir):
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
    # TODO len(entities) can be zero, then no comparison is needed
    dataset = {}
    sentence_tuples = []
    # process each file in directory
    for f in os.listdir(datadir):
        # parse XML file , obtaining a DOM tree
        tree = parse(datadir + f)
        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes["id"].value  # get sentence id
            stext = s.attributes["text"].value  # get sentence text
            # tokenize text
            tokens = tokenize(stext)
            entities = s.getElementsByTagName("entity")
            B_tag = True
            #for each word tuple, check if any entity tag contains the word and type
            for word_tuple in tokens:
                found_type = False
                for e in entities:
                    if e.attributes["text"].value == word_tuple[0]:
                        if B_tag == True:
                            new_tuple = word_tuple + ('B-' + e.attributes["type"].value,)
                        else:
                            new_tuple = word_tuple + ('I-' + e.attributes["type"].value,)
                        sentence_tuples.append(new_tuple)
                        found_type=True
                        #break to not add double
                        break
                #type not found, set to None
                if found_type == False:
                    new_tuple = word_tuple + ('O',)
                    sentence_tuples.append(new_tuple)
                dataset[sid] = sentence_tuples
                
                B_tag = False

    return dataset

    #Use XML parsing and tokenization functions from previous exercises

def create_indexs(dataset, max_length):
    '''
    Task:   Create index dictionaries both for input (words) and output (labels)
            from given dataset .

    Input:  dataset:    dataset produced by load_data.
            max_length: maximum length of a sentence (longer sentences will
                        be cut, shorter ones will be padded).

    Output: A dictionary where each key is an index name (e.g. " words ", " labels ") ,
            and the value is a dictionary mapping each word / label to a number .
            An entry with the value for maxlen is also stored.

    Example:
        >>> create_indexs(traindata)
            {’words’: { ’<PAD > ’:0 , ’<UNK >’:1 , ’11-day’:2 , ’murine’:3 , ’criteria’:4 ,
            ’stroke ’:5 ,... ,’ levodopa’:8511 , ’terfenadine’: 8512}
            ’labels ’: {’<PAD > ’:0 , ’B- group ’:1 , ’B- drug_n ’:2 , ’I- drug_n ’:3 , ’O ’:4 ,
            ’I- group ’:5 , ’B- drug ’:6 , ’I- drug ’:7 , ’B- brand ’:8 , ’I- brand ’:9}
            ’maxlen ’ : 100
            }
    '''

    idx = {}
    words_dict = {}
    labels_dict = {}

    words_dict['<PAD>'] = 0
    words_dict['<UNK>'] = 1
    labels_dict['<PAD>'] = 0
    labels_dict['O'] = 1
    word_counter = 2
    label_counter = 2

    #iterate sentences
    for key,value in dataset.items():
        #iterate list of tuples
        for word_tuple in value:
            #check if word has a type and if key already exists
            if (word_tuple[3] != 'O') and (word_tuple[0] not in words_dict.keys()):
                words_dict[word_tuple[0]] = word_counter
                word_counter = word_counter + 1
                
                #check if label key already exists
                if word_tuple[3] not in labels_dict.keys():
                    labels_dict[word_tuple[3]] = label_counter
                    label_counter = label_counter + 1


    idx['words'] = words_dict
    idx['labels'] = labels_dict
    idx['maxlen'] = max_length

    return idx

    #Add a ’<PAD>’:0 code to both ’words’ and ’labels’ indexes.
    #Add an ’<UNK>’:1 code to ’words’.
    #The coding of the rest of the words/labels is arbitrary.
    #This indexes will be needed by the predictor to properly use the model.

def encode_words(dataset, idx):
    '''
    Task: Encode the words in a sentence dataset formed by lists of tokens into
    lists of indexes suitable for NN input .

    Input:  dataset:    A dataset produced by load_data .
            idx:        A dictionary produced by create_indexs, containing word and
                        label indexes , as well as the maximum sentence length .

    Output: The dataset encoded as a list of sentence , each of them is a list of
            word indices . If the word is not in the index , <UNK> code is used . If
            the sentence is shorter than max_len it is padded with <PAD> code .

    Example:
        >>> encode_words(traindata, idx)
            [[6882 1049 4911 ... 0 0 0]
            [2290 7548 8069 ... 0 0 0]
            ...
            [2002 6582 7518 ... 0 0 0]]
    '''
    words_encoded = []
    maxlen = idx['maxlen']

    # iterate sentences
    for key, value in dataset.items():
        sentence = []
        iterations = 1
        # iterate list of tuples (word, start, end, type)
        for word_tuple in value:
            #c heck if word in dict, else unknown UNK tag (1 directly instead of doing a lookup)
            if word_tuple[0] in idx['words']:
                sentence.append(idx['words'][word_tuple[0]])
            else:
                sentence.append(1)
            if iterations == maxlen:
                break
            iterations = iterations + 1

        # sentence shorter than maxlen, add padding
        if len(value) < maxlen:
            sentence = (sentence + maxlen * [0])[:maxlen]

        words_encoded.append(np.array(sentence))
        
    return np.array(words_encoded)


def encode_labels(dataset, idx):
    '''
    Task:   Encode the ground truth labels in a sentence dataset formed by lists of
            tokens into lists of indexes suitable for NN output.

    Input:  dataset:    A dataset produced by load_data.
            idx:        A dictionary produced by create_indexs, containing word and
                        label indexes, as well as the maximum sentence length.

    Output: The dataset encoded as a list of sentence, each of them is a list of
            BIO label indices. If the sentence is shorter than max_len it is
            padded with <PAD > code .

    Example:
        >>> encode_labels (traindata, idx)
            [[[4] [6] [4] [4] [4] [4] ... [0] [0]]
            [[4] [4] [8] [4] [6] [4] ... [0] [0]]
            ...
            [[4] [8] [9] [4] [4] [4] ... [0] [0]]]
    '''
    labels_encoded = []
    maxlen = idx['maxlen']

    # iterate sentences
    for key, value in dataset.items():
        sentence = []
        iterations = 1
        # iterate list of tuples (word, start, end, type)
        for word_tuple in value:
            sentence.append(idx['labels'][word_tuple[3]])
            #labels_encoded.append(idx['labels'][word_tuple[3]])
            
            if iterations == maxlen:
                break
            iterations = iterations + 1

        # sentence shorter than maxlen, add padding
        if len(value) < maxlen:
            sentence = (sentence + maxlen * [0])[:maxlen]
                
        labels_encoded.append(np.array(sentence))

    return np.array(labels_encoded)
    #Note: The shape of the produced list may need to be adjusted depending
    #on the architecture of your network and the kind of output layer you use.


def output_entities(dataset, preds, outfile):
    '''
    Task: Output detected entities in the format expected by the evaluator

    Input:  dataset:    A dataset produced by load_data .
            preds:      For each sentence in dataset, a list with the labels for each
                        sentence token, as predicted by the model

    Output: prints the detected entities to stdout in the format required by the evaluator .

    Example:
        >>> output \ _entities (dataset, preds)
            DDI - DrugBank . d283 .s4 |14 -35| bile acid sequestrants | group
            DDI - DrugBank . d283 .s4 |99 -104| tricor | group
            DDI - DrugBank . d283 .s5 |22 -33| cyclosporine | drug
            DDI - DrugBank . d283 .s5 |196 -208| fibrate drugs | group
            ...
    '''
    #TODO preds might need further unpacking
    for key, value in dataset.items():
        sid = key
        # iterate list of tuples (word, start, end, type)
        for word_tuple, label in zip(value, preds):
            if label == 'O':
                print("not predicted as anything")
                continue
            #TODO check if this is the right way to calculate offset
            offset = word_tuple[1] + "-" + word_tuple[2]
            with open(outfile, 'a') as f:
                print(sid + "|" + offset + "|" + word_tuple[0] + "|" + label, file=f)
                
    #Note: Most of this function can be reused from NER-ML exercise.
