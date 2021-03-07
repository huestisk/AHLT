def extract_features(s):
    '''
    Task: Given a tokenized sentence, return a feature vector for each token
    
    Input:
        s: A tokenized sentence (list of triples (word, offsetFrom, offsetTo) )
        
    Output:
        A list of feature vectors, one per token. Features are binary and vectors 
        are in sparse representation (i.e. onlyactive features are listed)
        
    Example:>>> extract_features([(" Ascorbic ",0,7), ("acid ",9,12), (",",13,13),
        (" aspirin ",15,21), (",",22,22), ("and",24,26), ("the",28,30), (" common ",32,37), 
        ("cold ",39,42), (".",43,43)])[ [ "form=Ascorbic",  "suf4=rbic", "next=acid", 
        "prev=_BoS_", "capitalized"  ],[ "form=acid",  "suf4=acid", "next=,", "prev=Ascorbic"],
        [ "form=,",  "suf4=,", "next=aspirin", "prev=acid", "punct" ],[ "form=aspirin",  
        "suf4=irin", "next=,", "prev=," ], ...]
    '''

    pass