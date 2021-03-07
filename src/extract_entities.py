
drugbank = './resources/DrugBank.txt'
hsbd = './resources/HSDB.txt'

drugs = []
brands = []
groups = []

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
	# TODO: presence of numbers, partial capitalization
	if word in drugs:
		return "drug"
	elif word in brands:
		return "brand"
	elif word in groups:
		return "group"
	elif before =='(' and after == ')':
		return "drug"
	elif before =='[' and after == ']':
		return "drug"
	elif word.isupper():
		return "brand"
	elif word[-5:] in ['azole', 'idine', 'amine', 'mycin']:
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

	# TODO: Add n-grams

	for idx, token in enumerate(s):

		word = token[0]
		before = s[idx-1][0] if idx != 0 else None
		after = s[idx-1][0] if idx != len(s)-1 else None 

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
