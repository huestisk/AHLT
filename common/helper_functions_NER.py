# import nltk
# nltk.download('punkt')

import os
import re
import random
import numpy as np

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
specialchars = ('(', ')', '[', ']', '*', '+', '?')


def tokenize(s):
	"""
	Task: Given a sentence, calls nltk.tokenize to split it into tokens, and adds to each token its start/end offsetin the original sentence.
	Input:
		s: string containing the text for one sentence
	Output:
		Returns a list of tuples (word , offsetFrom , offsetTo)
	Example:
		> tokenize (" Ascorbic acid , aspirin , and the commoncold .")
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


def extract_features(s):
    """
    Task: Given a tokenized sentence, return a feature vector for each token
    Input:
        s: A tokenized sentence (list of triples (word, offsetFrom, offsetTo))
    Output:
        A list of feature vectors, one per token. Features are binary and vectors 
        are in sparse representation (i.e. onlyactive features are listed)
    Example:
        >>> extract_features([(" Ascorbic ",0,7), ("acid ",9,12), (",",13,13),
        (" aspirin ",15,21), (",",22,22), ("and",24,26), ("the",28,30), (" common ",32,37), 
        ("cold ",39,42), (".",43,43)])
        
        [["form=Ascorbic",  "suf4=rbic", "next=acid", "prev=_BoS_", "capitalized"],
         ["form=acid",  "suf4=acid", "next=,", "prev=Ascorbic"],
         ["form=,", "suf4=,", "next=aspirin", "prev=acid", "punct"],
         ["form=aspirin", "suf4=irin", "next=,", "prev=,"], ...]
    """

    tokens = [i for i,j,k in s]
    bias = ['bias'] * len(tokens)
    # full token
    f1 = ["form=" + token.lower() for token in tokens]
    # last four characters
    f2 = ["suf4=" + token[-4:].lower() for token in tokens]
    # next token
    f3 = ["next=" + token.lower() for token in tokens]
    f3 = f3[1:] + ["next=_EoS_"]
    # previous token
    f4 = ["prev=" + token.lower() for token in tokens]
    f4 = ["prev=_BoS_"] + f4[:-1]
    # last three characters
    f5 = ["suf3=" + token[-3:].lower() for token in tokens]
    # last five characters
    f6 = ["suf5=" + token[-5:].lower() for token in tokens]
    # uppercase
    f7 = ["upper=%s" % token.isupper() for token in tokens]
    # are there numbers
    f8 = ["digit=%s" % any((char.isdigit() for char in token)) for token in tokens]
    # last 4 next token
    f9 = ["-4next=" + token[-4:].lower() for token in tokens]
    f9 = f9[1:] + ["-4next=_EoS_"]
    # last 4 previous token
    f10 = ["-4prev=" + token[-4:].lower() for token in tokens]
    f10 = ["-4prev=_BoS_"] + f10[:-1]

    return list(zip(bias, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10))


def get_tag(token, gold):
    """
    Task: Given a token and a list of ground truth entites in a sentence, decide 
        which is the B-I-O tag for the token
    Input:
        token: A token, i.e. one triple (word, offsetFrom, offsetTo)
        gold: A list of ground truth entities, i.e. a list of triples 
            (offsetFrom, offsetTo, type)  
    Output: 
        The B-I-O ground truth tag for the given token ("B-drug", "I-drug", 
        "B-group", "I-group", "O", ...)
    Example:
        >>> get_tag ((" Ascorbic ",0,7), [(0, 12, "drug"), (15, 21, "brand")])
        B-drug
        >>> get_tag ((" acid",9,12), [(0, 12, "drug"), (15, 21, "brand ")])
        I-drug
        >>> get_tag ((" common ",32,37), [(0, 12, "drug"), (15, 21, "brand")])
        O
        >>> get_tag ((" aspirin ",15,21), [(0, 12, "drug"), (15, 21, "brand ")])
        B-brand
    """
    for truth in gold:
        if token[1] == truth[0] and token[2] <= truth[1]:
            return "B-" + truth[2]
        elif token[1] > truth[0] and token[2] <= truth[1]:
            return "I-" + truth[2]
    return "0"


drugbank = './resources/DrugBank.txt'
hsbd = './resources/HSDB.txt'

brands = []
groups = []

drug_suffixes = ['racin', 'NaFlu', 'teine', 'butin', 'ampin', 'navir', 'azone', 'ncers', 'moter', 'orine', 'limus', 'ytoin', 'angin', 'ucose', 'sulin', 'odone', 'xacin', 'udine', 'osine', 'zepam', 'hacin', 'etine', 'pride', 'ridol', 'apine', 'idone', 'apine', 'nafil', 'otine', 'oxide', 'iacin', 'tatin', 'cline', 'kacin', 'xacin', 'illin',
				 'azole', 'idine', 'amine', 'mycin', 'tatin', 'ridin', 'caine', 'micin', 'hanol', 'dolac', 'feine', 'lline', 'amide', 'afine', 'rafin', 'trast', 'goxin', 'coxib', 'phine', 'coids', 'isone', 'oride', 'ricin', 'lipin', 'cohol', 'otine', 'taxel', 'tinib', 'rbose', 'ipine', 'idine', 'nolol', 'uride', 'lurea', 'ormin', 'amide', 'estin']
group_suffixes = [s[1:] + "s" for s in drug_suffixes] + ['trate', 'otics', 'pioid', 'zones'] # plural of drugs can be group names

with open(hsbd, 'r') as f:
	drugs = f.read().splitlines()

with open(drugbank, 'r') as f:
	for line in f.readlines():
		raw = line.split('|')
		if raw[1] == "drug\n":
			drugs.append(raw[0])
		elif raw[1] == "brand\n":
			brands.append(raw[0])
		elif raw[1] == "group\n":
			groups.append(raw[0])


def get_entity_type(word, before, after):
	label = None
	# if word in groups:
	# 	label = "group"
	# elif word in drugs:
	# 	label = "drug"
	# elif word in brands:
	# 	label = "brand"
	if word.isupper() and word[-1] == 's':
		label = "group"
	elif word.isupper():
		label = "brand" if random.random() > 0.66 else "drug"
	elif word[-5:] in group_suffixes:
		label = "group"
	elif word[-5:] in drug_suffixes:
		label = "drug"
	elif sum((char.isdigit() for char in word)) > 1: # at least two numbers
		label = "drug"
	elif '-' in word and any(char.isdigit() for char in word):
		label = "drug"
	elif before == '(' and after == ')' or before == '[' and after == ']':
		label = "drug"

	if label == "drug" and random.random() < 0.04: # some should be labeled drug_n
		label = "drug_n"
	
	return label


# def get_entity_type_bigram(word, before, after_bigram):
# 	if word in drugs:
# 		return "drug"
# 	elif word in brands:
# 		return "brand"
# 	elif word in groups:
# 		return "group"


def extract_entities(s):
	"""Task: Given a tokenized sentence, identify which tokens (or groups of consecutive tokens) are drugs
	Input:
		s: A tokenized sentence (list of triples (word, offsetFrom, offsetTo) )
	Output:
		A list of entities. Each entity is a dictionary with the keys ’name ’, ’offset ’, and ’type ’.
	Example:
	>>> extract_entities ([(" Ascorbic ",0,7), ("acid ",9,12), (",",13,13), ("aspirin ",15,21), (",",22,22), ("and",24,26), ("the",28,30), (" common",32,37), ("cold ",39,42), (". ,43,43)])
	[{" name ":" Ascorbic acid", "offset ":"0-12", "type ":"drug"}, {"name ":" aspirin", "offset ":"15-21", "type ": "brand"}]
	"""
	entities = []
	# save_bigram = False
	for idx, token in enumerate(s):
		# if save_bigram:
		# 	entity = {
		# 		"text": bigram,
		# 		"offset": str(start) + "-" + str(token[2]),
		# 		"type": entity_type
		# 	}

		# 	entities.append(entity)
		# 	save_bigram = False
		# 	continue

		word = token[0]
		before = s[idx-1][0] if idx != 0 else None
		after = s[idx+1][0] if idx != len(s)-1 else None

		# if after is not None:
		# 	bigram = word + ' ' + after
		# 	after_bigram = s[idx+2][0] if idx != len(s)-2 else None
		# 	entity_type = get_entity_type_bigram(word, before, after_bigram)
		# 	if entity_type is not None:
		# 		save_bigram = True
		# 		start = token[1]
		# 		continue

		entity_type = get_entity_type(word, before, after)

		if entity_type is None:
			continue

		entity = {
			"text": word,
			"offset": str(token[1]) + "-" + str(token[2]),
			"type": entity_type
		}

		entities.append(entity)

	return entities


""""""""""""" Lab 5 """""""""""""

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
    # iterate over all tokens
    counter = 2
    for _, value in dataset.items():
        for word_tuple in value:
            if not word_tuple[0] in words.keys():
                words[word_tuple[0]] = counter
                counter += 1

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
            if idx >= len(y_sen):
                break
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


