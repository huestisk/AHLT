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



