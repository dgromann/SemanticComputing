# Model solutions for the Tutorial 1 of the lecture on Semantic Computing
# TU Dresden, SS 2018, Dagmar Gromann

print("Solutions Exercise 1: see lecture slides")
print()

# Exercise 2:
print("Solutions Exercise 2: ")
print("2a) ")
print("12/(4+1) =", 12/(4+1), "; 3*3 =", 3*3, "; [\"Semantic\", \"Computing\"]*5 =",["Semantic", "Computing"]*5)

from nltk.book import *

print("2b) ")
print("Number of words in Text 2: ", len(text2), "Number of words in Text 2: ", len(set(text2)))
print("2c) ")
print("Lexical Diversity: ", len(set(text2))/len(text2))
print("2d) ")
print("Text2 collocations: ", text2.collocations())
print("Text6 collocations: ", text6.collocations())
print("Collocation is a group of words that occur together in text unusually often.")
print("2e) ")
print("Semantic Computing[9:18] = ", "Semantic Computing"[9:18], "; Semantic Computing[9:] = ", "Semantic Computing"[9:], "; Semantic Computing[-9:] =", "Semantic Computing"[-9:], "; [\"Semantic\", \"Computing\"][1] =", ["Semantic", "Computing"][1])
print("Explanation: The first three expressions slice a string, while the last expression accesses the index of a list. "
      "When slicing a string and no end index is provided, e.g. [9:], than the end of the whole string is used as end index. "
      "Minus 9 gives us the last nine characters of the string if used in the beginning index, e.g. [-9:].")
print("2f) ")
print("sent1[2][2]: ", sent1[2][2], " is the third letter (index 2) of the third word (index 2) of the first sentence in the NLTK books corpus.")
print()


#Exercise 3:
print("Solution Exercise 3: Lesk algorithm")

from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer

ps = PorterStemmer()


text = open("text.txt", "r")
eval = open("evaluation.txt", "r")

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


results = dict()
for result in eval.readlines()[1:]:
    results[result.split(";")[1]] = result.split(";")[2]

corr_results = 0
lines = []
for line in text.readlines()[1:]:
    lines.append(line)
    word = line.split(";")[0]
    sentence = line.split(";")[3]

    line_result = lesk(sentence, word)
    print(results[line.split(";")[1]].strip(), line_result.strip())

    if line_result and (results[line.split(";")[1]].strip() == line_result.strip()):
       corr_results += 1

print("Correct results: ", corr_results)
print("Percentage: ", corr_results/len(results))

#Exercise 4
print("Solution Exercise 4:")
print("The results are printed above. Some of the main problems are wrong part-of-speech or selection of a neighboring sense. \n"
      "Only taking the context of the sentence and the definitions of one specific word into account is not enough as the accuracy shows.")

#Exercise 5
print("Solution Exercise 5:")


def lesk_extended(sentence, ambiguous_word, pos=None, hyperhypo=True):
    max_overlaps = 0
    lesk_dictionary = []
    lesk_sense = ""
    context = sentence.split()

    for sense in wn.synsets(ambiguous_word):

        if pos and sense.pos() is not pos:
            continue

        if hyperhypo == True:
            for i in sense.hypernyms() + sense.hyponyms():
                lesk_dictionary += i.lemma_names()

        lesk_dictionary += sense.definition().split()
        lesk_dictionary += sense.lemma_names()

        overlaps = set(lesk_dictionary).intersection(context)

        if len(overlaps) > max_overlaps:
            lesk_sense = sense
            max_overlaps = len(overlaps)
    return str(lesk_sense)

corr_results_ext = 0
for line in lines:
    word = line.split(";")[0]
    sentence = line.split(";")[3]

    line_result = lesk_extended(sentence, word, pos=line.split(";")[2])
    #print(results[line.split(";")[1]].strip(), line_result.strip())

    if line_result and (results[line.split(";")[1]].strip() == line_result.strip()):
       corr_results_ext += 1

print(corr_results_ext)
print("Percentage: ", corr_results_ext/len(results))