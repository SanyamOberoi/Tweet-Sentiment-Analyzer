from flask import Flask, render_template, request, redirect, url_for
# import tensorflow as tf
# from keras.models import load_model
from sentiment_model import clean_text, lemmatizer_func, create_model
import numpy as np
import dill
import gzip

app = Flask(__name__)

@app.route('/')
def main():
	return redirect('/index')

@app.route('/index', methods=["GET"])
def index():
	return render_template('index.html')

@app.route('/predict', methods=["GET","POST"])
def predict():

	if request.method == "GET":
		tweet = request.args.get("tweet")
	elif request.method == "POST":
		tweet = request.form['text']

	with gzip.open('model/RandomForest_balanced_acc.dill.gz','rb') as f:
		pipe = dill.load(f)
	
	# with open('model/tokenizer.dill','rb') as f:
	# 	tokenizer = dill.load(f)

	with open('model/label_encoder.dill','rb') as f:
		label_encoder = dill.load(f)
	
	# pipe = create_model(tokenizer)

	predicted_sentiment = pipe.predict([tweet])
	predicted_sentiment_class = label_encoder.inverse_transform(predicted_sentiment)[0]
	predicted_sentiment_class_prob = pipe.predict_proba([tweet])[0,predicted_sentiment][0]

	res = "Predicted Tweet Sentiment Class: {}. Predicted Class Probability: {:.2f}".format(predicted_sentiment_class, predicted_sentiment_class_prob)

	return render_template('result.html', tweet= tweet, result=res)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=80) # , debug=True