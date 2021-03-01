from nltk.tokenize import word_tokenize
import re

# import nltk
# nltk.download('punkt')

specialchars = ('(', ')', '[', ']', '*', '+', '?')

def tokenize(s):
	"""
	Task: 
		Given a sentence, calls nltk.tokenize to split it into tokens, and adds to each token its start/end offsetin the original sentence.

	Input:
		s: string containing the text for one sentence

	Output:
		Returns a list of tuples (word , offsetFrom , offsetTo)
		
		Example:>>> tokenize (" Ascorbic acid , aspirin , and the commoncold .")
		
		[(" Ascorbic ",0,7), ("acid",9,12), (",",13,13), ("aspirin ",15,21), (",",22,22), ("and",24,26), ("the",28,30), (" common ",32,37), ("cold ",39,42),("." ,43 ,43)]
	"""

	tokens0 = word_tokenize(s)
	# tokens = [tuple([word]) + re.search(word, s).regs[0] for word in tokens0]

	tokens = []
	shift = 0
	for word in tokens0:
		if word.startswith(specialchars): word = "\\" + word # FIXME
		ans = re.search(word, s[shift:])
		if ans is None: continue
		start = ans.start() + shift
		shift += ans.end()
		tokens.append((word, start, shift-1))

	return tokens
