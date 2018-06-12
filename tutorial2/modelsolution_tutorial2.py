# TU Dresden, SS 2018, Dagmar Gromann
# Model solutions for the Tutorial 2 of the lecture on Semantic Computing

import nltk
from nltk.sem.logic import *

read_expr = Expression.fromstring
logic_parser = LogicParser()

#Exercise 1:
from nltk.sem.drt import *
dexpr = DrtExpression.fromstring
sentence = "Anna owns a dog and it bit me."
drs1 = dexpr('([x,y][anna(x),dog(y),own(x,y)]) -> ([z][me(z),bite(y,z)])')
print("Solution Exercise 1: ")
print(drs1.fol())
print(drs1.pretty_format())
print()

#Eercise 2:
from nltk.tokenize import word_tokenize
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import wordnet as wn
tagger = PerceptronTagger()

sentence = "It just tears me apart to see you suffering like that and in tears."
pos_tags = tagger.tag(word_tokenize(sentence))
print("Solution Exercise 1: ")
print(word_tokenize(sentence))
print(pos_tags)
for ss in wn.synsets('tear'):
     print(ss, ss.definition())
print("tear.v.02 is the verb in the sentence and sounds like t-air, while tear.n.01 is the noun in the sentence and sounds like t-ear.")
print()

#Exercise 3:

from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer

ps = PorterStemmer()
ls = LancasterStemmer()
ss = SnowballStemmer("english")

print("Solution Exercise 3:")
print("Input words: presumably, provisions, owed, cement")
print("Porter Stemmer: ", ps.stem("presumably"), ps.stem("provisions"), ps.stem("owed"), ps.stem("abacus"))
print("Lancaster Stemmer: ", ls.stem("presumably"), ls.stem("provisions"), ls.stem("owed"), ls.stem("abacus"))
print("Snowball Stemmer: ", ss.stem("presumably"), ss.stem("provisions"), ss.stem("owed"), ss.stem("abacus"))

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

print("Lemmatizer: ", lemmatizer.lemmatize("presumably"), lemmatizer.lemmatize("provisions"), lemmatizer.lemmatize("owed"), lemmatizer.lemmatize("abacus"))
print()

#Exercise 4:
from nltk.corpus import gutenberg
print("Solution Exercise 4:")
print("All the text files in the corpus: ", gutenberg.fileids())

text = gutenberg.words("shakespeare-macbeth.txt")
text_vocab = [w.lower() for w in text if w.isalpha()]
lemma = set(lemmatizer.lemmatize(t) for t in text_vocab)
stems = set(ps.stem(t) for t in text_vocab)

print("Length of the text: ", len(text))
print("Length of the words in the text in lowercase and without punctuation and numbers: ", len(text_vocab))
print("Lemma count: ", len(lemma), "Stems count: ", len(stems))
print("Lemmas listed: ", lemma)
print("Stems listed: ", stems)

filtered = [w for w in lemma if len(wn.synsets(w)) > 0]
print("Filtered lemmas using WordNet: ", filtered)
print("Length of filtered lemmas list: ", len(filtered))

not_filtered = [x for x in text_vocab if x not in filtered]
print("Words that are presumably not in WordNet: ",not_filtered)

#For comparison here the WordNet filtering on the initial vocbulary
filtered2 = [w for w in set(text_vocab) if len(wn.synsets(w))>0]
print("Length of filtered input vocabulary using WordNet: ", len(filtered2))