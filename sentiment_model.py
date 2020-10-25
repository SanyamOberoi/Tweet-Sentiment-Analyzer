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
# from nltk import TweetTokenizer
# from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
# import spacy
# from spacy.lang.en import STOP_WORDS
# import gensim.downloader as api

# import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from keras.utils import to_categorical
# from keras import Input

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import Pipeline
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

# from scipy.stats import uniform,randint
from numpy import asarray, zeros
 
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


def data_normalize(text):
	cleaned_text = clean_text(text)
	return ' '.join([lemmatizer.lemmatize(word) for word in cleaned_text.split()])

def data_prep(data_list, tokenizer, max_len):
	data = tokenizer.texts_to_sequences(data_list)
	data = pad_sequences(data, padding='post', maxlen=max_len)
	return data


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

def create_model(vocab_size, embed_dim, max_len, embedding_matrix):
	lstm_out1 = 64
	lstm_out2 = 32

	model = Sequential()
	model.add(Embedding(vocab_size, embed_dim, weights=[embedding_matrix], input_length = max_len, trainable=False, name='embedding_layer')) 
	model.add(Bidirectional(LSTM(lstm_out1, return_sequences=True, dropout=0.5),merge_mode='concat', name='bi_lstm_layer1'))
	model.add(Bidirectional(LSTM(lstm_out2), merge_mode='concat', name='bi_lstm_layer2'))
	model.add(Dense(3,activation='softmax', name='dense'))
	return model

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

	# df_train['cleaned_text'] = df_train['text'].apply(lambda x: clean_text(x))
	df_train['normalized_cleaned_text'] = df_train['text'].apply(lambda x: data_normalize(x))

	# df_test['cleaned_text'] = df_test['text'].apply(lambda x: clean_text(x))
	df_test['normalized_cleaned_text'] = df_test['text'].apply(lambda x: data_normalize(x))

	# Vectorizing text data
	# Unigram Vectorizer had ~8800 unique words after removing stop words and having min freq = 2
	max_features = 5000
	tokenizer = Tokenizer(num_words=max_features)
	tokenizer.fit_on_texts(df_train['normalized_cleaned_text'])

	X_train = tokenizer.texts_to_sequences(df_train['normalized_cleaned_text'])
	max_len = max([len(x) for x in X_train])
	X_train = pad_sequences(X_train, padding='post', maxlen=max_len)

	X_test = data_prep(df_test['normalized_cleaned_text'], tokenizer, max_len)

	# Label Encoding the target variable
	label_encoder = LabelEncoder()
	y_train = label_encoder.fit_transform(df_train['sentiment'])
	y_test = label_encoder.transform(df_test['sentiment'])

	# Turning output into one-hot encoded vectors
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)

	# Persisting label_encoder for use at prediction time
	with open('model/label_encoder.dill','wb') as f:
		dill.dump(label_encoder, f)

	vocab_size = len(tokenizer.word_index) + 1
	embed_dim = 200


	# embedding_matrix = fetch_embeddings(tokenizer, vocab_size, embed_dim)
	with gzip.open('model/embedding_matrix.dill.gz','rb') as f:
		embedding_matrix = dill.load(f) 
	
	model = create_model(vocab_size, embed_dim, max_len, embedding_matrix)
	model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

	# print(model.summary())

	history = model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2) # , class_weight=

	score = model.evaluate(X_test, y_test, verbose=1)

	print("Test Score:", score[0])
	print("Test Accuracy:", score[1])

	# Persisting tokenizer
	with open('model/tokenizer.dill','wb') as f:
		dill.dump(tokenizer, f)

	# Persisting model 
	model.save('model/bi-lstm')
