# TU Dresden, SS 2018, Dagmar Gromann
# Basic lesk to be extended in tutorial 1

from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def lesk(sentence, ambiguous_word):
    max_overlaps = 0
    lesk_dictionary = []
    lesk_sense = ""
    context = sentence.split()

    for sense in wn.synsets(ambiguous_word):
        lesk_dictionary += sense.definition().split()
        lesk_dictionary += sense.lemma_names()

        overlaps = set(lesk_dictionary).intersection(context)

        if len(overlaps) > max_overlaps:
            lesk_sense = sense
            max_overlaps = len(overlaps)
    return str(lesk_sense)