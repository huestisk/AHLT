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
        subprocess.Popen([command], cwd=working_directory,
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


def yes_or_no(question):
    reply = str(input(question + ' (y/N): ')).lower().strip()
    return True if reply[0] == 'y' else False


def getInfo(datadir):
    filename = datadir.split('/')
    filename = filename[0] + '/dependency_trees_' + filename[1] + '.pkl'
    try:
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
    except (FileNotFoundError, EOFError):
        if not yes_or_no("Could not find data. Reparse?"):
            raise Exception
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
        # Case: elem ends before arr starts
        elem_end_to_arr_start = int(elem[1]) - int(arr[0])
        if elem_end_to_arr_start < 0:
            continue
        # Case: elem starts after arr ends
        elem_start_to_arr_end = int(elem[0]) - int(arr[1])
        if elem_start_to_arr_end > 0:
            continue
        elem_len = int(elem[1]) - int(elem[0])
        elem_start_to_arr_start = int(elem[0]) - int(arr[0])
        elem_end_to_arr_end = int(elem[1]) - int(arr[1])
        # Case: complete overlap (elem smaller than arr)
        if elem_start_to_arr_start >= 0 and elem_end_to_arr_end <= 0:
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
    # idx = arr_in_list(arr, list_arrays)
    # if idx is not None:
    #     return idx
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


clues_effect    = ['effect', 'potentiate', 'enhance', 'report', 'inhibitor', 'include',
                   'inhibit', 'cause', 'augment', 'diminish', 'affect']
clues_mechanism = ['increase', 'reduce', 'produce', 'show', 'reduction', 'lower', 
                   'decrease', 'inhibit', 'interfere', 'have', 'alter', 'impair']
clues_int       = ['interact', 'interaction', 'include']
clues_advise    = ['exceed', 'co-administration', 'use', 'co-administer', 
                   'administer', 'avoid', 'consider', 'administraion']


def check_clues(node):
    lemma = node['lemma']
    if lemma in clues_effect:
        return 'effect'
    elif lemma in clues_mechanism:
        return 'mechanism'
    elif lemma in clues_int:
        return 'int'
    elif lemma in clues_advise:
        return 'advise'
    else:
        return None


def check_interaction(analysis, entities, e1, e2, stext=None):
    """
    Task: Decide whether a sentence is expressing a DDI between two drugs.

    Input:  analysis: a DependencyGraph object with all sentence information
            entites: A list of all entities in the sentence(id and offsets)
            e1, e2: ids of the two entities to be checked.

    Output: Returns the type of interaction (’effect’,’mechanism’,’advice’,’int’) between e1 and e2 expressed by the sentence, or ’None’ if no interaction is described.
    """

    result = None

    # Extract Node offsets
    nodes = analysis.nodes
    n_offsets = [np.array((nodes[idx]['start'], nodes[idx]['end'])).astype(
        str) for idx in nodes if idx is not None]

    # Get node IDs for entities
    ids = dict()
    for e in [e1, e2]:
        e_offset = np.array(entities[e])
        n_idx = find_best_node_match(e_offset, n_offsets)
        n_key = list(nodes.keys())[n_idx]
        if n_key != 0:
            ids[e] = nodes[n_key]['address']
        else:
            # FIXME: match not found because dependency tree is faulty...
            # Faulty documents: d15.s0, d68.s0, d4.s0, d134.s0
            return None     # raise Exception("Match not found.")

    id_e1 = ids[e1]
    id_e2 = ids[e2]

    # Build subtree
    iterations = 0
    subtree = [[id_e1], [id_e2]]
    while iterations < len(nodes):
        # Update branch 1
        node_e1 = nodes[subtree[0][-1]]['head']
        if node_e1 != 0:
            subtree[0].append(node_e1)
        # Update branch 2
        node_e2 = nodes[subtree[1][-1]]['head']
        if node_e2 != 0:
            subtree[1].append(node_e2)
        iterations += 1
        # Check if tree complete
        if subtree[0][-1] == subtree[1][-1]:
            break
        elif subtree[0][-1] in subtree[1]:  # Prune
            idx = subtree[1].index(subtree[0][-1])
            subtree[1] = subtree[1][:idx+1]
            break
        elif subtree[1][-1] in subtree[0]:  # Prune
            idx = subtree[0].index(subtree[1][-1])
            subtree[0] = subtree[0][:idx+1]
            break

    # rule 0 - check if branch lengths are too big
    branch_length = [len(branch) for branch in subtree]
    # if max(branch_length) > 5 or min(branch_length) > 3:
    #    return None

    # rule 1 - check whether one entity is inside the subject of one
    # verb, and the other is inside the direct object of the same verb
    lowest_common_subsummer = nodes[subtree[0][-1]]
    if lowest_common_subsummer['ctag'].startswith('VB') and min(branch_length) != 1:
        rel1 = analysis.nodes[subtree[0][-2]]['rel']
        rel2 = analysis.nodes[subtree[1][-2]]['rel']
        if (rel1 == 'nsubj' and rel2 == 'obj') or (rel2 == 'nsubj' and rel1 == 'obj'):
            result = check_clues(lowest_common_subsummer)

    if result is not None:
        return result

    # rule 2 - check if there is a verb between nodes
    for between in range(id_e1+1, id_e2):
        node = analysis.nodes[between]
        if node['ctag'].startswith('VB'):
            result = check_clues(node)

    return result


""" ML Functions """


def computeFeatures(analysis, entities, e1, e2, stext):
    # Extract Node offsets
    nodes = analysis.nodes
    n_offsets = [np.array((nodes[idx]['start'], nodes[idx]['end'])).astype(
        str) for idx in nodes if idx is not None]

    # Get node IDs for entities
    ids = dict()
    for e in entities:
        e_offset = np.array(entities[e])
        n_idx = find_best_node_match(e_offset, n_offsets)
        n_key = list(nodes.keys())[n_idx]
        if n_key != 0:
            ids[e] = nodes[n_key]['address']
        else:
            # FIXME: match not found because dependency tree is faulty...
            # Faulty documents: d15.s0, d68.s0, d4.s0, d134.s0
            return None     # raise Exception("Match not found.")

    id_e1 = ids[e1]
    id_e2 = ids[e2]

    # Build subtree
    iterations = 0
    subtree = [[id_e1], [id_e2]]
    while iterations < len(nodes):
        # Update branch 1
        node_e1 = nodes[subtree[0][-1]]['head']
        if node_e1 != 0:
            subtree[0].append(node_e1)
        # Update branch 2
        node_e2 = nodes[subtree[1][-1]]['head']
        if node_e2 != 0:
            subtree[1].append(node_e2)
        iterations += 1
        # Check if tree complete
        if subtree[0][-1] == subtree[1][-1]:
            break
        elif subtree[0][-1] in subtree[1]:  # Prune
            idx = subtree[1].index(subtree[0][-1])
            subtree[1] = subtree[1][:idx+1]
            break
        elif subtree[1][-1] in subtree[0]:  # Prune
            idx = subtree[0].index(subtree[1][-1])
            subtree[0] = subtree[0][:idx+1]
            break

    # Interesting varibales
    branch_length = [len(branch) for branch in subtree]
    least_common_subsummer = nodes[subtree[0][-1]]
    lcs_in_list = check_clues(least_common_subsummer)
    
    lcs_id = subtree[0][-1]
    if (lcs_id > id_e1 and lcs_id < id_e2) or (lcs_id > id_e2 and lcs_id < id_e1):
        lcs_loc = 'between'
    elif lcs_id < id_e1 and lcs_id < id_e2:
        lcs_loc = 'before'
    elif lcs_id > id_e1 and lcs_id > id_e2:
        lcs_loc = 'after'
    else:
        lcs_loc = None

    between = None
    bet_lemma = None
    for node_id in range(id_e1+1, id_e2):
        node = analysis.nodes[node_id]
        if node['ctag'].startswith('VB'):
            between = check_clues(node)
            if between is not None:
                bet_lemma = node['lemma']
                break

    path = ''
    path_lemma = ''
    for b in subtree[0]:
        if b == subtree[0][-1]:
            path += nodes[b]['tag']
        else:
            path += nodes[b]['rel'] + "<"
    for b in reversed(subtree[1]):
        if b != subtree[1][-1]:
            path += ">" + nodes[b]['rel']

    if len(subtree[0]) > 1:
        short_path = str(nodes[subtree[0][-2]]['rel']) + "<"
    else:
        short_path = "None<"
    
    short_path += str(least_common_subsummer['tag']) + ">"

    if len(subtree[1]) > 1:
        short_path += str(nodes[subtree[1][-2]]['rel']) 
    else:
        short_path += "None"

    ents_on_path = False
    ents_between = False
    for id in ids.values():
        if id != id_e1 and id != id_e2:
            if not ents_on_path:
                ents_on_path = id in subtree[0] or id in subtree[1]
            if not ents_between:
                ents_between = (id > id_e1 and id < id_e2) or (id > id_e2 and id < id_e1)

    # Create features
    features = []
    features.append(f"e1_tag={nodes[id_e1]['tag']}")
    features.append(f"e2_tag={nodes[id_e2]['tag']}")
    features.append(f"e1_lemma={nodes[id_e1]['lemma']}")
    features.append(f"e2_lemma={nodes[id_e2]['lemma']}")

    features.append(f"lcs_tag={least_common_subsummer['tag']}") 
    features.append(f"lcs_lemma={least_common_subsummer['lemma']}")
    features.append(f"lcs_list={lcs_in_list}")
    features.append(f"lcs_loc={lcs_loc}")

    features.append(f"branch1_long={branch_length[0] > 3}")
    features.append(f"branch2_long={branch_length[1] > 3}")
    
    features.append(f"bet_list={between}")
    features.append(f"bet_lemma={bet_lemma}")
    features.append(f"direct_above={min(branch_length) == 1}")

    features.append(f"other_ents={len(entities) > 2}")
    features.append(f"ents_on_path={ents_on_path}")
    features.append(f"ents_between={ents_between}")

    features.append(f"path={path}")
    features.append(f"short_path={short_path}")

    return features


def getFeatures(outfile, datadir, recompute=False):
    # delete old file
    if recompute and os.path.exists(outfile):
        os.remove(outfile)
    # load features
    try:
        features = pickle.load(open(outfile, 'rb'))
        return features
    except FileNotFoundError:
        if recompute:
            pass
        elif not yes_or_no("Could not load features. Recompute?"):
            raise Exception
    # Recompute the features
    features = []
    # Load dependency trees
    info = getInfo(datadir)
    # process each file in directory
    for f in os.listdir(datadir):
        # parse XML file , obtaining a DOM tree
        tree = parse(datadir + "/" + f)
        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            # we're only considering pairs
            pairs = s.getElementsByTagName("pair")
            if len(pairs) <= 0:
                continue
            # get sentence id
            sid = s.attributes["id"].value
            stext = s.attributes["text"].value
            # get dependency tree
            analysis = info[sid].getDependencyGraph()
            # load sentence entities into a dictionary
            entities = {}
            for e in s.getElementsByTagName("entity"):
                entities[e.attributes["id"].value] = e.attributes["charOffset"].value.split("-")
            # get DDI
            for p in pairs:
                id_e1 = p.attributes["e1"].value
                id_e2 = p.attributes["e2"].value
                gold = None
                if p.hasAttribute("type"):
                    gold = p.attributes["type"].value
                # Compute features
                data = np.array((sid, id_e1, id_e2, gold), dtype=str)
                feats = computeFeatures(
                    analysis, entities, id_e1, id_e2, stext)
                if feats is None:
                    continue  # FIXME
                data = np.concatenate((data, feats))
                features.append(data)

    # Write to file
    features = np.array(features, dtype=object)
    pickle.dump(features, open(outfile, 'wb'))

    return features




""""""""""""" Lab 6 """""""""""""
from helper_functions_NER import tokenize

def get_sentence_tokens(text: str, entities: dict, e1: str, e2: str) -> list:
    '''
    Compute sentence tokens by parsing and masking entities.
    '''
    tokens = tokenize(text)
    # extract token offsets
    t_offsets = [np.array((token[1:])) for token in tokens]
    tokens = [token[0] for token in tokens]

    # match entities to token
    for e in entities.keys():
        e_offset = np.array(entities[e])
        t_idx = find_best_node_match(e_offset, t_offsets)
        if e == e1:
            tokens[t_idx] = '<DRUG1>'
        elif e == e2:
            tokens[t_idx] = '<DRUG2>'
        else:
            tokens[t_idx] = '<DRUG_OTHER>'

    return tokens


def get_tree_tokens(analysis, entities: dict, e1: str, e2: str) -> list:

    # Extract Node offsets
    nodes = analysis.nodes
    n_offsets = [np.array((nodes[idx]['start'], nodes[idx]['end'])).astype(
        str) for idx in nodes if idx is not None]

    # Get node IDs for entities
    ids = dict()
    for e in entities:
        e_offset = np.array(entities[e])
        n_idx = find_best_node_match(e_offset, n_offsets)
        n_key = list(nodes.keys())[n_idx]
        if n_key != 0:
            ids[e] = nodes[n_key]['address']
        else:
            # FIXME: match not found because dependency tree is faulty...
            # Faulty documents: d15.s0, d68.s0, d4.s0, d134.s0
            return None     # raise Exception("Match not found.")

    # convert to token
    tokens = [(nodes[i]['word'], nodes[i]['lemma'], nodes[i]['tag']) for i in range(len(nodes))]

    for key, id in ids.items():
        if key == e1:
            tokens[id] = ('<DRUG1>', '<DRUG1>', '<DRUG1>')
        elif key == e2:
            tokens[id] = ('<DRUG2>', '<DRUG2>', '<DRUG2>')
        else:
            tokens[id] = ('<DRUG_OTHER>', '<DRUG_OTHER>', '<DRUG_OTHER>')

    return tokens[1:]


def load_data(datadir, full_parse=False):
    '''
    Task:   Load XML files in given directory, tokenize each sentence, and extract
            learning examples (tokenized sentence + entity pair).

    Input:  datadir: A directory containing XML files.

    Output: A list of classification cases. Each case is a list containing sentence
            id, entity1 id, entity2 id, ground truth relation label, and a list
            of sentence tokens (each token containing any needed information: word,
            lemma, PoS, offsets, etc).

    Example:
        >>> load_data(’data/Train’)
            [[’DDI-DrugBank.d66.s0’, DDI-DrugBank.d66.s0.e0’, ’DDI-DrugBank.d66.s0.e1’, ’null’,
             [(’<DRUG1>’, ’<DRUG1>’, ’<DRUG1>’), (’-’, ’-’, ’:’),
              (’Concomitant’, ’concomitant’,’JJ’), (’use’,’use’,’NN’),
              (’of’, ’of’, ’IN’), (’<DRUG2>’, ’<DRUG2>’, ’<DRUG2>’), (’and’, ’and’, ’CC’),
              (’<DRUG_OTHER>’, ’<DRUG_OTHER>’, ’<DRUG_OTHER>’),
              (’may’, ’may’, ’MD’),
               ..., 
              (’syndrome’, ’syndrome’, ’NN’), (’.’, ’.’, ’.’)
            ]]
            ...
             [’DDI-MedLine.d94.s12’, ’DDI-MedLine.d94.s12.e1’, ’DDI-MedLine.d94.s12.e2’, ’effect’,
             [(’The’, ’the’, ’DT’), (’uptake’, ’uptake’, ’NN’),
              (’inhibitors’, ’inhibitor ’, ’NNS’),
              (’<DRUG_OTHER>’, ’<DRUG_OTHER>’, ’<DRUG_OTHER>’) , (’and’, ’and’, ’CC’),
              (’<DRUG1>’,’ <DRUG1>’,’ <DRUG1> ’) ,
               ...,
              (’effects’, ’effect’, ’NNS’), (’of’, ’of’, ’IN’),
              (’<DRUG2>’, ’<DRUG2>’, ’<DRUG2>’), (’in’, ’in’, ’IN’), ...
            ]]
            ...
    '''
    classification_cases = []
    data_type = datadir.split('/')[1]
    filename = 'data/DDI_{}_'.format(data_type) 
    filename += 'full_parse.pkl' if full_parse else '.pkl'

    try:
        with open(filename, 'rb') as f:
            classification_cases = pickle.load(f)
            return classification_cases
    except (FileNotFoundError, EOFError):
        if not yes_or_no("Could not find DDI data. Reparse?"):
            raise Exception

    # parse as dependency tree
    if full_parse:
        info = getInfo(datadir)
    
    for f in os.listdir(datadir):
        if not f.endswith('.xml'):
            continue
        # parse XML file , obtaining a DOM tree
        tree = parse(datadir + f)
        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            # get sentence id
            sid = s.attributes["id"].value
            # get sentence text
            text = s.attributes["text"].value
            # we're only considering pairs
            pairs = s.getElementsByTagName("pair")

            # load sentence entities into a dictionary
            if len(pairs) > 0:
                # get dependency tree
                if full_parse:
                    analysis = info[sid].getDependencyGraph()
                # get entities
                entities = {}
                ents = s.getElementsByTagName("entity")
                for e in ents:
                    eid = e.attributes["id"].value
                    entities[eid] = e.attributes["charOffset"].value.split("-")
                # save tokens
                for p in pairs:
                    e1 = p.attributes['e1'].value
                    e2 = p.attributes['e2'].value
                    # if ddi exist, get type
                    gold = 'null'
                    if p.attributes["ddi"].value == 'true':
                        gold = p.attributes["type"].value
                    
                    # tokenize text
                    if full_parse:
                        tokens = get_tree_tokens(analysis, entities, e1, e2)
                    else:
                        # parse only words (not lemma, PoS)
                        tokens = get_sentence_tokens(text, entities, e1, e2)

                    case = [sid, e1, e2, gold, tokens]
                    classification_cases.append(case)

    # save parse
    with open(filename, 'wb') as f:
        pickle.dump(classification_cases, file=f)

    return classification_cases


def create_indices(dataset, max_length):
    '''
    Task:   Create index dictionaries both for input (words) and output (labels)
            from given dataset.

    Input:  dataset: dataset produced by load_data.
            max_length: maximum length of a sentence (longer sentences will
            be cut, shorter ones will be padded).

    Output: A dictionary where each key is an index name (e.g. "words", "labels"),
            and the value is a dictionary mapping each word/label to a number.
            An entry with the value for maxlen is also stored
    Example:
        >>> create_indexs(traindata)
            {’words’: {’<PAD>’:0, ’<UNK>’:1, ’11-day’:2, ’murine’:3, ’criteria’:4,
            ’stroke’:5,...,’levodopa’:8511, ’terfenadine’:8512}
            ’labels’: {’null’:0, ’mechanism’:1, ’advise’:2, ’effect’:3, ’int’:4}
            ’maxlen’: 100
            }
    '''
    words = {
        '<PAD>': 0,
        '<UNK>': 1
    }

    def add_to_words(token, counter: int) -> int:
        if isinstance(token, str) and not token in words.keys():
            words[token] = counter
            counter += 1
        elif isinstance(token, tuple):
            for t in token:
                counter = add_to_words(t, counter)
        elif not isinstance(token, str):
            raise RuntimeError

        return counter

    # iterate over all tokens
    counter = 2
    for ddi in dataset:
        if ddi[4] is None:      # TODO: Count how many are corrupt
            continue
        for token in ddi[4]:
            counter = add_to_words(token, counter)

    labels = {
        'null': 0,
        'mechanism': 1,
        'advise': 2,
        'effect': 3,
        'int': 4
    }

    idx = {
        'words': words,
        'labels': labels,
        'maxlen': max_length
    }

    return idx


def encode(dataset, idx):
    num_tokens = len(dataset[0][4][0])
    words_encoded = np.zeros((len(dataset), idx['maxlen'] * num_tokens))
    labels_encoded = np.zeros((len(dataset),))

    # iterate sentences
    for i, item in enumerate(dataset):
        if item[4] is None:
            continue
        for j, word in enumerate(item[4]):
            for k, token in enumerate(word):
                if token in idx['words'].keys():
                    words_encoded[i, j*num_tokens + k] = idx['words'][token]
                else:
                    words_encoded[i, j*num_tokens + k] = 1      # Word unknown
            # shorten long sentences
            if j >= idx['maxlen'] - 1:
                break
        # encode label
        labels_encoded[i] = idx['labels'][item[3]]

    return words_encoded, labels_encoded


def output_entities(dataset, preds, outfile) -> None:
    '''
    Task: Output detected DDIs in the format expected by the evaluator

    Input:
        dataset: A dataset produced by load_data.
        preds: For each sentence in dataset, a label for its DDI type (or ’null’ if no DDI detected)

    Output:
        prints the detected interactions to stdout in the format required by the evaluator.

    Example:
        >>> output_interactions(dataset , preds) 
            DDI-DrugBank.d398.s0|DDI-DrugBank.d398.s0.e0|DDI-DrugBank.d398.s0.e1|effect 
            DDI-DrugBank.d398.s0|DDI-DrugBank.d398.s0.e0|DDI-DrugBank.d398.s0.e2|effect 
            DDI-DrugBank.d211.s2|DDI-DrugBank.d211.s2.e0|DDI-DrugBank.d211.s2.e5|mechanism 
            ...
    '''
    for y_pred, data in zip(preds, dataset):
        with open(outfile, 'a') as f:
            print(data[0] + "|" + data[1] + "|" + data[2] + "|" + y_pred, file=f)
