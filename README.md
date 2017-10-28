# Reminder-Extraction
A Natural Language Processing tool to extract reminder chunks out of text.

The particular project attempts to solve a small problem in the domain of natural language chunking and tagging.
Basically a Maxent model has been trained to chunk phrases in a body of text which are indicating the presence of a reminder.


## Basic Usage
- Clone the repository
- Run extract_reminder.py as python extract_reminder.py (This might take some time as the models are loaded.)
- Enter some text
- Wait for output

```
extractor=ReminderTagger(use_pretrained=True)



while True:
    sentence=raw_input("Enter text: ")
    print "Reminder : ",extractor.predict(sentence)
  
 ```
 
## Example:
> Enter text: Remind me to sleep at 3.


> Reminder :  sleep.


> Enter text: remind me to eat my medicines.


> Reminder :  eat my medicines.



 
