from nltk import tokenize


def tokenize(s):
    """
    Task: 
		Given a sentence, calls nltk.tokenize to split it intokens, and adds to each token its start/end offsetin the original sentence.

  	Input:
  		s: string containing the text for one sentence

	Output:
		Returns a list of tuples (word , offsetFrom , offsetTo)Example:>>> tokenize (" Ascorbic acid , aspirin , and the commoncold .")[(" Ascorbic ",0,7), ("acid",9,12), (",",13,13), ("aspirin ",15,21), (",",22,22), ("and",24,26), ("the",28,30), (" common ",32,37), ("cold ",39,42),("." ,43 ,43)]
	"""
	pass
