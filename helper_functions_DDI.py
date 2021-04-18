import os
import subprocess

working_directory = os.getcwd() + '/stanford-corenlp-4.2.0/'
command = 'java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer'
p = subprocess.Popen([command], cwd=working_directory,
                     shell=True)  # Run the server


import re
from nltk.parse.corenlp import CoreNLPDependencyParser

my_parser = CoreNLPDependencyParser(url="http://localhost:9000")

specialchars = ('(', ')', '[', ']', '*', '+', '?')  # FIXME: hack

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
    if not s:
        return None

    input = re.sub('%', '%25', s)

    # parse text (as many times as needed)
    mytree, = my_parser.raw_parse(input)

    shift = 0
    for idx in range(len(mytree.nodes)):
        node = mytree.nodes[idx]
        word = node['word']

        if word is None:
            continue
        elif word.startswith(specialchars): 
            word = "\\" + word # FIXME: hack

        ans = re.search(word, s[shift:])
        if ans is not None:
            node['start'] = ans.start() + shift
            shift += ans.end()
            node['end'] = shift - 1

    return mytree


def check_interaction(analysis, entities, e1, e2): 
    """
    Task: Decide whether a sentence is expressing a DDI between two drugs.

    Input:  analysis: a DependencyGraph object with all sentence information
            entites: A list of all entities in the sentence(id and offsets)
            e1, e2: ids of the two entities to be checked.

    Output: Returns the type of interaction(’effect’,’mechanism’,’advice’,’int’) between e1 and e2 expressed by the sentence, or ’None’ if no interaction is described.
    """
    clues_effect = ['administer', 'potentiate', 'prevent']
    clues_mechanism = ['reduce', 'increase', 'decrease']
    clues_int = ['interact', 'interaction']

    #ids for entities
    id_e1 = int(e1[-1])
    id_e2 = int(e2[-1])

    # Get offsets for entities (not used yet)
    for e in entities:
        if e == e1:
            e1_offset = entities[e]
        elif e == e2:
            e2_offset = entities[e]


    if len(entities) > 2:
        #print("***********")
        #print("analysis: ", len(analysis.nodes))
        #print("e1 :", id_e1, "e2 :", id_e2)
        #print("len entities:", len(entities))

        # check words between e1 and e2, naive but a first step for comparison
        for idx in range(id_e1+1, id_e2):
            node = analysis.nodes[idx]
            word = node['word']

            # print(node)

            if word in clues_effect:
                return 'effect'
            elif word in clues_mechanism:
                return 'mechanism'
            elif word in clues_int:
                return 'int'

    return 'effect'
