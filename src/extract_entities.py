

def extract_entities(s):
	"""Task:
		Given a tokenized sentence, identify which tokens (or groups ofconsecutive tokens) are drugs
		
	Input:
		s: A tokenized sentence (list of triples (word, offsetFrom, offsetTo) )
		
	Output:
		A list of entities. Each entity is a dictionary with the keys ’name ’, ’offset ’, and ’type ’.
		
		Example:
		>>> extract_entities ([(" Ascorbic ",0,7), ("acid ",9,12), (",",13,13), ("aspirin ",15,21), (",",22,22), ("and",24,26), ("the",28,30), (" common",32,37), ("cold ",39,42), (". ,43,43)])
		
		[{" name ":" Ascorbic acid", "offset ":"0-12", "type ":" drug"},{"name ":" aspirin", "offset ":"15 -21", "type ":" brand "}]
	"""

	pass
