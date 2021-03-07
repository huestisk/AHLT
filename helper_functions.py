# import nltk
# nltk.download('punkt')

from nltk.tokenize import word_tokenize
specialchars = ('(', ')', '[', ']', '*', '+', '?')

def tokenize(s):
	'''
	Task: 
		Given a sentence, calls nltk.tokenize to split it into tokens, and adds to each token its start/end offsetin the original sentence.

	Input:
		s: string containing the text for one sentence

	Output:
		Returns a list of tuples (word , offsetFrom , offsetTo)
		
    Example:
        >>> tokenize (" Ascorbic acid , aspirin , and the commoncold .")
		[(" Ascorbic ",0,7), ("acid",9,12), (",",13,13), ("aspirin ",15,21), 
        (",",22,22), ("and",24,26), ("the",28,30), (" common ",32,37), ("cold ",39,42),("." ,43 ,43)]
	'''

	tokens0 = word_tokenize(s)

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



def extract_features(s):
    '''
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
    '''

    tokens = [i for i,j,k in s]
    # full token
    f1 = ["form=" + token for token in tokens]
    # last four characters
    f2 = ["suf4=" + token[-4:] for token in tokens]
    # next token
    f3 = ["next=" + token for token in tokens]
    f3 = f3[1:] + ["next=_EoS_"]
    # previous token
    f4 = ["prev=" + token for token in tokens]
    f4 = ["prev=_BoS_"] + f4[:-1]
    
    return list(zip(f1, f2, f3, f4))



def get_tag(token, gold):
    '''
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
    '''

    pass