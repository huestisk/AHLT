import os
import re
import subprocess
import pickle

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
                    data_dict[sid] = tree   #(tree.tree_string, tree.starts, tree.ends)
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    return data_dict


def check_interaction(analysis, entities, e1, e2):
    """
    Task: Decide whether a sentence is expressing a DDI between two drugs.

    Input:  analysis: a DependencyGraph object with all sentence information
            entites: A list of all entities in the sentence(id and offsets)
            e1, e2: ids of the two entities to be checked.

    Output: Returns the type of interaction(’effect’,’mechanism’,’advice’,’int’) between e1 and e2 expressed by the sentence, or ’None’ if no interaction is described.
    """
    clues_effect = ['administer', 'potentiate', 'prevent', 'administers',
                    'potentiates', 'prevents', 'effect', 'effects', 'reaction', 'reactions']
    clues_mechanism = ['reduce', 'increase', 'decrease', 'reduces',
                       'increases', 'increased', 'decreases', 'decreased']
    clues_int = ['interact', 'interaction', 'interacts', 'interactions']
    clues_advise = ['should', 'recommended']

    # ids for entities
    id_e1 = int(e1[-1])
    id_e2 = int(e2[-1])

    # Get offsets for entities (not used yet)
    # Are these of use here?
    for e in entities:
        if e == e1:
            e1_offset = entities[e]
        elif e == e2:
            e2_offset = entities[e]

    # for i in reversed(range(id_e2)):
    node_e1 = analysis.nodes[id_e1]
    node_e2 = analysis.nodes[id_e2]
    head_e1 = node_e1['head']
    head_e2 = node_e2['head']

    # TODO What type of interaction should be returned here
    if head_e2 == head_e1:
        #print("UNDER THE SAME WORD")
        node = analysis.nodes[id_e2]
        if node['ctag'] == 'VB':
            #print("UNDER THE SAME VERB")
            return 'int'

    # TODO What type of interaction should be returned here

    # Check if e1 or e2 is over/under each other
    # TODO iterate further heads
    # TODO What type of interaction should be returned here
    elif head_e1 == id_e2:
        #print("E2 IS OVER E1")
        pass
    elif head_e2 == id_e1:
        #print("E1 IS OVER E2")
        pass

    if len(entities) > 2:

        # check words between e1 and e2, naive but a first step for comparison
        for idx in range(id_e1+1, id_e2):
            node = analysis.nodes[idx]
            word = node['word']
            lemma = node['lemma']

            # print(node)

            if word or lemma in clues_effect:
                return 'effect'
            elif word or lemma in clues_mechanism:
                return 'mechanism'
            elif word or lemma in clues_int:
                return 'int'
            elif word or lemma in clues_advise:
                return 'advise'

    next_node_e1 = analysis.nodes[id_e1 + 1]
    next_node_e2 = analysis.nodes[id_e2 + 1]

    if next_node_e1['word'] or next_node_e2['word'] in clues_advise:
        return 'advise'

    return None
