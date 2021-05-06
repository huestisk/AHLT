# import nltk
# nltk.download('punkt')

import re
import random
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
