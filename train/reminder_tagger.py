import nltk
from pickle import load
from pickle import dump
from maxent_tagger import MaxentTagger
from bigram_tagger import BigramTagger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import SnowballStemmer
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from pickle import load
import pandas as pd

class ReminderTagger(object):
    def __init__(self,use_pretrained=False,save_model=True,tagger="maxent_tagger"):
        self.use_pretrained=use_pretrained
        self.save_model=save_model
        features_file=open('./trained_models/bow_features.pkl','rb')
        self.features = load(features_file)
        features_file.close()
        if(use_pretrained==True):
            cv_file=open('./trained_models/count_vectorizer.pkl','rb')
            self.cv=load(cv_file)
            cv_file.close()
            tf_file=open('./trained_models/tf_idftransformer.pkl','rb')
            self.tf=load(tf_file)
            tf_file.close()
            clf_file=open('./trained_models/reminder_classifier.pkl','rb')
            self.clf=load(clf_file)
            clf_file.close()
            tagger_file=open('./trained_models/tagger.pkl','rb')
            self.tagger=load(tagger_file)
            tagger_file.close()





        else:
            self.cv=None
            self.tf=None
            self.clf=None
            self.tagger_flag=tagger







    def data_preparation(self,text,reminders):
        tagged_text=[nltk.pos_tag(t.split()) for t in text]
        print tagged_text[0]

        train_sents=[]
        for i in range(len(tagged_text)):
            train_sent=[]
            reminder_words = nltk.word_tokenize(reminders[i])
            for token in tagged_text[i]:
                if(token[0] in reminder_words):
                    train_sent.append(((token[0],token[1]),'REMIND'))
                else:
                    train_sent.append(((token[0], token[1]), 'O'))
            train_sents.append(train_sent)
        return train_sents

    def normalize_data(self,text):
        stemmer = SnowballStemmer("english")
        text = text.decode('utf-8', 'ignore')
        text = text.lower()
        text = re.sub(r"[^a-z0-9]", " ", text)
        words = nltk.word_tokenize(text)
        words = [stemmer.stem(w) for w in words]
        words = [w for w in words if w != ' ']
        words = [w for w in words if w != '']
        return " ".join(words)

    def train(self,text,reminders,save_models=True):
        if(self.use_pretrained==True):
            print "Use predict method to predict"
            return
        processed_text=[self.normalize_data(t) for t in text]
        self.cv=CountVectorizer()
        count_vectors=self.cv.fit_transform(processed_text)
        cv_file=open('./trained_models/count_vectorizer.pkl','wb')
        dump(self.cv,cv_file,-1)
        cv_file.close()
        self.tf=TfidfTransformer()
        X=self.tf.fit_transform(count_vectors).toarray()
        X=pd.DataFrame(X,columns=self.cv.get_feature_names())
        X=X[self.features].values
        tf_file=open("./trained_models/tf_idftransformer.pkl","wb")
        dump(self.tf,tf_file,-1)
        tf_file.close()
        y=np.zeros(len(reminders))
        for i in range(len(reminders)):
            if(reminders[i]!=""):
                y[i]=1

        self.clf=LogisticRegression()
        self.clf.fit(X,y)

        classifier_file=open('./trained_models/reminder_classifier.pkl','wb')
        dump(self.clf,classifier_file,-1);
        classifier_file.close()

        train_sents=self.data_preparation(text,reminders)
        print train_sents[0]
        if self.tagger_flag=='maxent_tagger':
            self.tagger=MaxentTagger(train_sents)
        else:
            self.tagger=BigramTagger(train_sents)

        tagger_file=open('./trained_models/tagger.pkl','wb')
        dump(self.tagger,tagger_file,-1)
        tagger_file.close()




    def predict(self,sentence):
        processed_sentence=self.normalize_data(sentence)
        cv_sentence=self.cv.transform([sentence])
        tf_sentence=self.tf.transform(cv_sentence)
        tf_df=pd.DataFrame(tf_sentence.toarray(),columns=self.cv.get_feature_names())
        tf_df=tf_df[self.features].values
        label=self.clf.predict(tf_df)
        if label==0:
            return "No Reminders Found"
        else:
            pos_tagged_sent=nltk.pos_tag(sentence.split())
            tagged_sent=self.tagger.tag(pos_tagged_sent)
            extracted=[w for((w,t),c) in tagged_sent if c=='REMIND']
            if len(extracted)==0:
                return "No Reminders Found"
            return " ".join(extracted)






























