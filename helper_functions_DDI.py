import os
import re
import subprocess
import pickle
import numpy as np

from xml.dom.minidom import parse
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse.dependencygraph import DependencyGraph

my_parser = None
specialchars = ('(', ')', '[', ']', '*', '+', '?')  # FIXME: hack


def connect_to_parser(run_parser=True):
    global my_parser
    if run_parser:
        working_directory = os.getcwd() + '/stanford-corenlp-4.2.0/'
        command = 'java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer'
        p = subprocess.Popen([command], cwd=working_directory,
                             shell=True)  # Run the server
    my_parser = CoreNLPDependencyParser(url="http://localhost:9000")


def analyze(s):
    """
    Task: Given one sentence, sends it to CoreNLP to obtain the tokens, tags, and dependency tree. 
    It also adds the start/end offsets to each token.

    Input: s: string containing the text for one sentence

    Output: Returns the nltk DependencyGraph (https://www.nltk.org/_modules/nltk/parse/dependencygraph.html) object produced by CoreNLP, enriched with token offsets.

    Example: analyze ("Caution should be exercised when combining resorcinol or salicylic acid with DIFFERIN Gel")

    {0: {’head’: None, ’lemma’: None, ’rel’: None, ’tag’: ’TOP’, ’word’: None}, 
     1: {’word’: ’Caution’, ’head’: 4, ’lemma’: ’caution’, ’rel’: ’nsubjpass’, ’tag’: ’NN’, ’start’: 0, ’end’: 6}, 
     2: {’word’: ’should’, ’head’: 4, ’lemma’: ’should’, ’rel’: ’aux’, ’tag’: ’MD’, ’start’: 8, ’end’: 13}, 
     3: {’word’: ’be’, ’head’: 4, ’lemma’: ’be’, ’rel’: ’auxpass’, ’tag’: ’VB’, ’start’: 15, ’end’: 16}, 
     4: {’word’: ’exercised’, ’head’: 0, ’lemma’: ’exercise’, ’rel’: ’ROOT’, ’tag’: ’VBN’, ’start’: 18, ’end’: 26}, 
     5: {’word’: ’when’, ’head’: 6, ’lemma’: ’when’, ’rel’: ’advmod’, ’tag’: ’WRB’, ’start’: 28, ’end’: 31}, 
     6: {’word’: ’combining’, ’head’: 4, ’lemma’: ’combine’, ’rel’: ’advcl’, ’tag’: ’VBG’, ’start’: 33, ’end’: 41}, 
     7: {’word’: ’resorcinol’, ’head’: 6, ’lemma’: ’resorcinol’, ’rel’: ’dobj’, ’tag’: ’NN, ...}
     ... }
    """
    if my_parser is None:
        connect_to_parser()
    if not s:
        return None

    input = re.sub('%', '%25', s)

    # parse text (as many times as needed)
    if my_parser is not None:
        mytree, = my_parser.raw_parse(input)
    else:
        raise Exception("Could not connect to parser.")

    shift = 0
    for idx in range(len(mytree.nodes)):
        node = mytree.nodes[idx]
        word = node['word']

        if word is None:
            continue
        elif word.startswith(specialchars):
            word = "\\" + word  # FIXME: hack

        ans = re.search(word, s[shift:])
        if ans is not None:
            node['start'] = ans.start() + shift
            shift += ans.end()
            node['end'] = shift - 1

    return mytree


class Tree():
    def __init__(self, analysis):
        self.tree_string = analysis.to_conll(10)
        self.starts = [analysis.nodes[i]['start'] if 'start' in analysis.nodes[i]
                       else None for i in range(len(analysis.nodes))]
        self.ends = [analysis.nodes[i]['end'] if 'end' in analysis.nodes[i]
                     else None for i in range(len(analysis.nodes))]

    def getDependencyGraph(self):
        # Convert from string
        dependencygraph = DependencyGraph(self.tree_string)
        # Add start & end values
        for i in range(len(dependencygraph.nodes)):
            dependencygraph.nodes[i]['start'] = self.starts[i]
            dependencygraph.nodes[i]['end'] = self.ends[i]

        return dependencygraph


def getInfo(datadir):
    filename = datadir.split('/')
    filename = filename[0] + '/dependency_trees_' + filename[1] + '.pkl'
    try:
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
    except (FileNotFoundError, EOFError) as e:
        data_dict = dict()
        connect_to_parser()
        # process each file in directory
        for f in os.listdir(datadir):
            if not f.endswith('.xml'):
                continue
            # parse XML file , obtaining a DOM tree
            tree = parse(datadir + "/" + f)
            # process each sentence in the file
            sentences = tree.getElementsByTagName("sentence")
            for s in sentences:
                # we're only considering pairs
                pairs = s.getElementsByTagName("pair")
                # save data
                if len(pairs) > 0:
                    # get sentence id
                    sid = s.attributes["id"].value
                    # Tokenize, tag, and parse sentence
                    analysis = analyze(s.attributes["text"].value)
                    tree = Tree(analysis)
                    # Save into dict
                    # (tree.tree_string, tree.starts, tree.ends)
                    data_dict[sid] = tree
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    return data_dict


def arr_in_list(arr, list_arrays):
    return next((idx for idx, elem in enumerate(list_arrays) if np.array_equal(elem, arr)), None)


def get_overlap(arr, list_arrays):
    arr_len = int(arr[1]) - int(arr[0])
    overlap = np.zeros((len(list_arrays),))
    # Get overlap for each element
    for i, elem in enumerate(list_arrays):
        if 'None' in elem:
            continue
        elem_len = int(elem[1]) - int(elem[0])
        elem_start_to_arr_start = int(elem[0]) - int(arr[0])
        elem_end_to_arr_start = int(elem[1]) - int(arr[0])
        elem_start_to_arr_end = int(elem[0]) - int(arr[1])
        elem_end_to_arr_end = int(elem[1]) - int(arr[1])
        # Case: elem ends before arr starts
        if elem_end_to_arr_start < 0:
            continue
        # Case: elem starts after arr ends
        elif elem_start_to_arr_end > 0:
            continue
        # Case: complete overlap (elem smaller than arr)
        elif elem_start_to_arr_start >= 0 and elem_end_to_arr_end <= 0:
            overlap[i] = min(arr_len, elem_len)
        # Case: complete overlap (elem larger than arr)
        elif elem_start_to_arr_start <= 0 and elem_end_to_arr_end >= 0:
            overlap[i] = min(arr_len, elem_len)
        # Case: end of elem overlaps
        elif elem_start_to_arr_start <= 0:
            overlap[i] = elem_end_to_arr_start
        # Case: start of elem overlaps
        elif elem_start_to_arr_end <= 0:
            overlap[i] = - elem_start_to_arr_end
        else:
            raise Exception("Overlap not found.")

    return overlap


def find_best_node_match(arr, list_arrays):
    # Check for perfect match
    idx = arr_in_list(arr, list_arrays)
    if idx is not None:
        return idx
    # Find node with most overlap
    try:
        overlap = get_overlap(arr, list_arrays)
    # Case: ['166', '176;185', '186;188', '195'] --> [['166', '176'],['185', '186'],['188', '195]]
    except ValueError:
        arrs = np.array(';'.join(arr).split(';'))
        arrs = arrs.reshape((2, int(len(arrs)/2)))
        # Get overlaps of each offsets
        overlap = np.zeros((len(list_arrays),))
        for arr in arrs:
            overlap += get_overlap(arr, list_arrays)

    return np.argmax(overlap)


def check_interaction(analysis, entities, e1, e2, stext=None):
    """
    Task: Decide whether a sentence is expressing a DDI between two drugs.

    Input:  analysis: a DependencyGraph object with all sentence information
            entites: A list of all entities in the sentence(id and offsets)
            e1, e2: ids of the two entities to be checked.

    Output: Returns the type of interaction (’effect’,’mechanism’,’advice’,’int’) between e1 and e2 expressed by the sentence, or ’None’ if no interaction is described.
    """
    clues_effect = ['administer', 'potentiate', 'prevent', 'administers',
                    'potentiates', 'prevents', 'effect', 'effects', 'reaction', 'reactions']
    clues_mechanism = ['reduce', 'increase', 'decrease', 'reduces',
                       'increases', 'increased', 'decreases', 'decreased']
    clues_int = ['interact', 'interaction', 'interacts', 'interactions']
    clues_advise = ['should', 'recommended']

    # DEBUG
    if stext is not None:
        e1_word = stext[int(entities[e1][0]):int(entities[e1][-1])+1]
        e2_word = stext[int(entities[e2][0]):int(entities[e2][-1])+1]
        # words = [analysis.nodes[node]['word'] for node in analysis.nodes]

    # Extract Node offsets
    nodes = analysis.nodes
    n_offsets = [np.array((nodes[idx]['start'], nodes[idx]['end'])).astype(
        str) for idx in nodes if idx is not None]

    ## Get node IDs for entities
    ids = dict()
    for e in [e1, e2]:
        e_offset = np.array(entities[e])
        n_idx = find_best_node_match(e_offset, n_offsets)
        n_key = list(nodes.keys())[n_idx]
        if n_key != 0:
            ids[e] = nodes[n_key]['address']
        else:
            return None     # FIXME: match not found because dependency tree is faulty...
            # raise Exception("Match not found.")

    id_e1 = ids[e1]
    id_e2 = ids[e2]

    ## Build subtree
    iterations = 0
    subtree = [[id_e1], [id_e2]]
    while subtree[0][-1] != subtree[1][-1] and iterations < len(nodes):
        # Update branch 1
        node_e1 = nodes[subtree[0][-1]]['head']
        if node_e1 != 0:
            subtree[0].append(node_e1)
        # Update branch 2
        node_e2 = nodes[subtree[1][-1]]['head']
        if node_e2 != 0:
            subtree[1].append(node_e2)
        iterations += 1
        if iterations == len(nodes):
            print(0)

    node_e1 = nodes[id_e1]
    node_e2 = nodes[id_e2]
    head_e1 = node_e1['head']
    head_e2 = node_e2['head']

    if head_e2 == head_e1:      # FIXME: improve
        #print("UNDER THE SAME WORD")
        head = nodes[head_e1]
        word = head['word']
        lemma = head['lemma']
        if (word in clues_effect) or (lemma in clues_effect):
            return 'effect'
        elif (word in clues_mechanism) or (lemma in clues_mechanism):
            return 'mechanism'
        elif (word in clues_int) or (lemma in clues_int):
            return 'int'
        elif (word in clues_advise) or (lemma in clues_advise):
            return 'advise'
        # # TODO: check based on type
        # if head['ctag'] == 'VB' or head['ctag'] == 'VBN':
        #     #print("UNDER THE SAME VERB")
        #     return 'int'
    # TODO: Check if e1 or e2 is over/under each other
    elif head_e1 == id_e2:
        #print("E2 IS OVER E1")
        pass
    elif head_e2 == id_e1:
        #print("E1 IS OVER E2")
        pass
    else:   # TODO: Find common head, even if it's further up
        lowest_common_subsummer = nodes[subtree[0][-1]]
        word = lowest_common_subsummer['word']
        lemma = lowest_common_subsummer['lemma']
        if (word in clues_effect) or (lemma in clues_effect):
            return 'effect'
        elif (word in clues_mechanism) or (lemma in clues_mechanism):
            return 'mechanism'
        elif (word in clues_int) or (lemma in clues_int):
            return 'int'
        elif (word in clues_advise) or (lemma in clues_advise):
            return 'advise'

    # TODO: check if entities are between the pair

    # next_node_e1 = nodes[id_e1 + 1]
    # next_node_e2 = nodes[id_e2 + 1]

    # if next_node_e1['word'] or next_node_e2['word'] in clues_advise:
    #     return 'advise'

    return None
