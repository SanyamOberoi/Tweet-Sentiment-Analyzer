from flask import Flask, render_template, request, redirect, url_for
# import tensorflow as tf
from keras.models import load_model
from sentiment_model import data_prep
import numpy as np
import dill

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

	model = load_model('model/bi-lstm')
	with open('model/tokenizer.dill','rb') as f:
		tokenizer = dill.load(f)

	with open('model/label_encoder.dill','rb') as f:
		label_encoder = dill.load(f)
	
	tweet_prepped = data_prep([tweet], tokenizer, max_len=30)

	# predicted_sentiment = np.argmax(model.predict(tf.expand_dims(tweet_prepped,0)), axis=-1)
	predicted_sentiment = np.argmax(model.predict(tweet_prepped), axis=-1)
	predicted_sentiment_class = label_encoder.inverse_transform(predicted_sentiment)[0]

	# predicted_sentiment_class_prob = model.predict(tf.expand_dims(tweet_prepped,0))[0,predicted_sentiment][0]
	predicted_sentiment_class_prob = model.predict(tweet_prepped)[0,predicted_sentiment][0]

	res = "Predicted Tweet Sentiment Class: {}. Predicted Class Probability: {:.2f}".format(predicted_sentiment_class, predicted_sentiment_class_prob)

	return render_template('result.html', tweet= tweet, result=res)

if __name__ == '__main__':
	app.run(host='0.0.0.0') # 