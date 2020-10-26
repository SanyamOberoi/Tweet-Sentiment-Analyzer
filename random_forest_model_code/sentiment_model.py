# Importing required packages
import pandas as pd
import numpy as np
import os
# import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# NLP specific imports
import nltk
import re
import string
# from nltk.corpus import stopwords
from nltk import TweetTokenizer
# from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.lang.en import STOP_WORDS
# import gensim.downloader as api

# import tensorflow as tf
# from tensorflow import keras
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, LSTM, Bidirectional
# from keras.utils import to_categorical
# from keras import Input

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
# from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

# from scipy.stats import uniform,randint
# from numpy import asarray, zeros
 
# from sklearn.model_selection import train_test_split

import gzip
import dill

from html.parser import HTMLParser


# Cleaning up the tweet text
# Regex pattern for web URLs (fetched from 'https://gist.github.com/gruber/8891611')
url_regex = r"(?i)\b((?:https?:(?:\/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b\/?(?!@)))"

html_parser = HTMLParser()

# Creating WordNetLemmatizer object 
lemmatizer = WordNetLemmatizer()

# Using spacy's STOP_WORD list instead of nltk's stopword list because of its more comprehensive nature 
STOP_WORDS_lemma = set([lemmatizer.lemmatize(word) for word in list(STOP_WORDS)]).union(set(string.punctuation) - {'!'})


emoticon_dict = {
    ":)": "happy",
    ":‑)": "happy",
    ":-]": "happy",
    ":-3": "happy",
    ":->": "happy",
    "8-)": "happy",
    ":-}": "happy",
    ":o)": "happy",
    ":c)": "happy",
    ":^)": "happy",
    "=]": "happy",
    "=)": "happy",
    "=O": "happy",
    "=P": "happy",
    "=D": "happy",
    "<3": "happy",
    ":-(": "sad",
    ":(": "sad",
    ":c": "sad",
    ":<": "sad",
    ":[": "sad",
    ">:[": "sad",
    ":{": "sad",
    ">:(": "sad",
    ":-c": "sad",
    ":-< ": "sad",
    ":-[": "sad",
    ":-||": "sad"}

# Tweet text cleaning function
def clean_text(text):
    # Stripping leading and trailing whitespaces from tweet and lowercasing it
    text = text.strip().lower()
    
    # Converting HTML entities
    text = html_parser.unescape(text)
    
    # Removing usernames
    no_username = re.sub(r'@[\w]*',' ', text)
        
    # Removing web URLs from tweets since they don't contribute to sentiment
    no_url_text = re.sub(url_regex, ' ', no_username)
    
    # Removing hashtags
    # no_hashtag_url_text = re.sub(r'#\w+',' ',no_url_text)
    
    # Replacing emoticons with corresponding happy/sad word
    for emoticon, word in emoticon_dict.items():
        # no_hashtag_url_text = no_hashtag_url_text.replace(emoticon, word)
        no_url_text = no_url_text.replace(emoticon, word)
    
    # Removing some punctuations that may not add value to sentiment of tweet (retaining '!', '<' (part of <3), '=' (part of =O, =P))
    no_punct = re.sub(r'[%s]' % re.escape(''.join(list(set(string.punctuation) - {'!'}))), ' ', no_url_text) # no_hashtag_url_text
    
    # Removing special characters and digits
    alpha_text = re.sub(r'[^a-zA-Z]',' ',no_punct)
    
    # Single character removal
    alpha_text = re.sub(r"\s+[a-zA-Z]\s+", ' ', alpha_text)
    
    # Removing multiple spaces
    text_cleaned = re.sub(r'\s+', ' ', alpha_text)
    
    return text_cleaned

# Lemmatizer function
def lemmatizer_func(text):
    return [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(text)]


# COMMENT: SINCE THE ORIGINAL PRE-TRAINED GLOVE VECTOR FILE WAS ABOUT 2 GB IN SIZE, IT COULDN'T BE UPLOADED USING FREE TIERED SERVER ACCOUNTS
# def fetch_embeddings(tokenizer, vocab_size, embed_dim=200):
# 	# Loading pre-trained GloVe embeddings (trained on twitter data)
# 	embeddings_dictionary = dict()
# 	glove_file = open('embeddings/glove-twitter-200.txt', encoding="utf8")

# 	for line in glove_file:
# 	    records = line.split()
# 	    word = records[0]
# 	    vector_dimensions = asarray(records[1:], dtype='float32')
# 	    embeddings_dictionary [word] = vector_dimensions
# 	glove_file.close()

# 	# Mapping word embedding vectors to corresponding tokenizer indices
# 	embedding_matrix = zeros((vocab_size, embed_dim))
# 	for word, index in tokenizer.word_index.items():
# 	    embedding_vector = embeddings_dictionary.get(word)
# 	    if embedding_vector is not None:
# 	        embedding_matrix[index] = embedding_vector

# 	return embedding_matrix

def create_model(tokenizer):
	
	vectorizer = TfidfVectorizer(strip_accents='unicode',
                            preprocessor=clean_text,
                            tokenizer=lemmatizer_func,
                            ngram_range=(1,3),
                            stop_words=STOP_WORDS_lemma, 
                            min_df = 2, max_df = 0.5)

	skf = StratifiedKFold(n_splits=5, random_state = 0)

	param_grid = {'n_estimators': [int(x) for x in np.linspace(400, 1000, num = 100)], # randint(50,250)
				'max_depth': [int(x) for x in np.linspace(25, 60, num = 5)], # randint(5,20)
				'min_samples_leaf': [int(x) for x in np.linspace(25, 80, num = 5)], # randint(50,200)
				'min_samples_split': [int(x) for x in np.linspace(80, 150, num = 10)]} # randint(100,250)

	rf = RandomForestClassifier(n_jobs=-1, random_state=0, class_weight='balanced_subsample')
	# grid_search = GridSearchCV(rf, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
	random_search = RandomizedSearchCV(rf, n_iter=250, param_distributions=param_grid, cv=skf, scoring='balanced_accuracy', n_jobs=-1, random_state=0)

	pipe_2 = Pipeline([("vectorizer",vectorizer), ("random_search",random_search)])

	return pipe_2

if __name__ == '__main__':
	# Reading in training + validation and test data
	df_train = pd.read_csv('data/train.csv')
	df_test = pd.read_csv('data/test.csv')

	# print("Train data shape:", df_train.shape)
	# print(df_train.info())

	# print("Test data shape:", df_test.shape)
	# print(df_test.info())

	# Dropping empty tweets
	df_train = df_train[(df_train['text'].notna()) & (df_train['text'] != '')]
	# df_train.shape

	# Checking the distribution of tweet sentiment
	# df_train['sentiment'].value_counts()

	# Creating TweetTokenizer object 
	tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)

	X_train = df_train['text']
	X_test = df_test['text']

	# y_train = df_train['sentiment'].map({'positive':'1', 'neutral':'0', 'negative': '-1'})
	# y_test = df_test['sentiment'].map({'positive':'1', 'neutral':'0', 'negative': '-1'})

	label_encoder = LabelEncoder()
	y_train = label_encoder.fit_transform(df_train['sentiment'])
	y_test = label_encoder.transform(df_test['sentiment'])

	pipe = create_model(tokenizer)

	pipe.fit(X_train,y_train)

	print(pipe.score(X_train,y_train))
	print(pipe.score(X_test,y_test))

	print(classification_report(y_true = y_test,y_pred = pipe.predict(X_test)))

	# Persisting label_encoder for use at prediction time
	with open('model/label_encoder.dill','wb') as f:
		dill.dump(label_encoder, f)

	# Persisting tokenizer
	with open('model/tokenizer.dill','wb') as f:
		dill.dump(tokenizer, f)

	# Persisting model
	with gzip.open('model/RandomForest_balanced_acc.dill.gz','wb') as f:
		dill.dump(pipe, f, recurse=True)
	
    # # Persisting model component
	# with open('model/vectorizer.dill','wb') as f:
	# 	dill.dump(pipe['vectorizer'], f)

    # # Persisting model component
	# with open('model/random_search.dill','wb') as f:
	# 	dill.dump(pipe['random_search'], f)