import random

drugbank = './resources/DrugBank.txt'
hsbd = './resources/HSDB.txt'

drugs = []
brands = []
groups = []

drug_suffixes = ['racin', 'NaFlu','teine', 'butin','ampin', 'navir', 'azone', 'ncers', 'moter', 'orine', 'limus', 'ytoin', 'angin', 'ucose', 'sulin', 'odone', 'xacin', 'udine', 'osine', 'zepam', 'hacin', 'etine', 'pride', 'ridol', 'apine', 'idone', 'apine', 'nafil', 'otine', 'oxide', 'iacin', 'tatin', 'cline', 'kacin','xacin', 'illin', 'azole', 'idine', 'amine', 'mycin', 'tatin', 'ridin', 'caine', 'micin', 'hanol', 'dolac', 'feine', 'lline', 'amide', 'afine', 'rafin', 'trast','goxin', 'coxib', 'phine', 'coids', 'isone', 'oride', 'ricin', 'lipin', 'cohol', 'otine', 'taxel', 'tinib', 'rbose', 'ipine', 'idine', 'nolol', 'uride', 'lurea', 'ormin', 'amide', 'estin']
# plural of drugs can be group names
group_suffixes = [s + "s" for s in drug_suffixes]
group_suffixes = [s[1:] for s in group_suffixes]
group_suffixes = group_suffixes + ['trate', 'otics', 'pioid', 'zones', ]

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
	if word in drugs:
		return "drug"
	elif word in brands:
		return "brand"
	elif word in groups:
		return "group"
	if word.isupper():
		return "brand" if random.random() > 0.5 else "drug"
	elif word[-5:] in drug_suffixes:
		return "drug"
	elif word[-5:] in group_suffixes:
		return "group"
	elif before =='(' and after == ')':
		return "drug"
	elif before =='[' and after == ']':
		return "drug"
	else:
		return None

def extract_entities(s):
	"""Task:
		Given a tokenized sentence, identify which tokens (or groups of consecutive tokens) are drugs
		
	Input:
		s: A tokenized sentence (list of triples (word, offsetFrom, offsetTo) )
		
	Output:
		A list of entities. Each entity is a dictionary with the keys ’name ’, ’offset ’, and ’type ’.
		
		Example:
		>>> extract_entities ([(" Ascorbic ",0,7), ("acid ",9,12), (",",13,13), ("aspirin ",15,21), (",",22,22), ("and",24,26), ("the",28,30), (" common",32,37), ("cold ",39,42), (". ,43,43)])
		
		[{" name ":" Ascorbic acid", "offset ":"0-12", "type ":"drug"}, {"name ":" aspirin", "offset ":"15-21", "type ": "brand"}]
	"""
	entities = []
	save_bigram = False
	for idx, token in enumerate(s):
		if save_bigram:
			entity = {
				"text": bigram,
				"offset": str(start) + "-" + str(token[2]),
				"type": entity_type
			}

			entities.append(entity)
			save_bigram = False
			continue

		word = token[0]
		before = s[idx-1][0] if idx != 0 else None
		after = s[idx+1][0] if idx != len(s)-1 else None

		# FIXME: Bigrams are saved even if only the second word is a drug
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
