import nltk
from nltk.stem import SnowballStemmer

class MaxentTagger(nltk.TaggerI):
    def __init__(self, train_sents):

        train_set = []
        for sent in train_sents:
            untagged_sent = nltk.tag.untag(sent)
            history = []
            for i, (word, tag) in enumerate(sent):
                feature_set = self.features(untagged_sent, i, history)
                train_set.append((feature_set, tag))
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(train_set, trace=1)

    def tag(self, sentence):
        history = []
        for (i, word) in enumerate(sentence):
            feature_set = self.features(sentence, i, history)
            tag = self.classifier.classify(feature_set)
            history.append(tag)
        return zip(sentence, history)

    def features(self, sentence, i, history):
        stemmer = SnowballStemmer("english")
        sentence=[(w.decode('utf-8','ignore'),t) for (w,t)in sentence]
        word, pos = sentence[i]
        word = word.lower()

        word = stemmer.stem(word)

        if i == 0:
            prevword, prevpos, prevtag = "<START>", "START>", "<START>"
        else:
            prevword, prevpos = sentence[i - 1]
            prevword = prevword.lower()
            prevword = stemmer.stem(prevword)
            prevtag = history[i - 1]

        if i == len(sentence) - 1:
            nextword, nextpos = "<END>", "<END>"
        else:
            nextword, nextpos = sentence[i + 1]

        return {"pos": pos, "word": word, "prevword": prevword, "prevpos": prevpos, "nextword": nextword,
                "nextpos": nextpos,
                "prevtag": prevtag}