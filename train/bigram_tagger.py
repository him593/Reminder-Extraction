import nltk

class BigramTagger(nltk.TaggerI):
    def __init__(self, train_sents):

        self.tagger = nltk.BigramTagger(train_sents)

    def tag(self, sentence):
        tagged_sentence = self.tagger.tag(sentence)
        return tagged_sentence

