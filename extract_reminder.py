import pandas as pd
import numpy as np
from train.reminder_tagger import ReminderTagger


"""
 To train a new model follow:
 extractor=ReminderTagger(use_pretrained_false,save_model=True)
 extractor.train(texts,reminders)
 #Make sure that where there are no reminders are marked as empty string."
 $Then you can run the model as:
 extractor.predict("remind me to go shopping")
 go shopping
 
"""


"""
 To just use the pretrained models, use the following template:"
"""

extractor=ReminderTagger(use_pretrained=True)



while True:
    sentence=raw_input("Enter text: ")
    print "Reminder : ",extractor.predict(sentence)