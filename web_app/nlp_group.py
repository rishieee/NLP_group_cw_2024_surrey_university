import logging
from datetime import datetime
from flask import Flask, jsonify, request
from flask_json import FlaskJSON, JsonError, json_response, as_json
from flask_restful import Resource, Api
import sqlite3
import sys
from process import *
from model_predict import *

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
	def get(self):
		return "Hello NLP!"

api.add_resource(HelloWorld, '/')


def save_user_imputs(time, text, prediction, prediction_time):

	connection = sqlite3.connect("db/user_input.db")
	cursor = connection.cursor()
	createTable = ''' CREATE TABLE IF NOT EXISTS PREDICTIONS (
		QueryDate TIMESTAMP,
		QueryText VARCHAR(400),
		Prediction VARCHAR(500),
		Prediction_time VARCHAR(100)
		); '''
	cursor.execute(createTable)

	insertQuery = """INSERT INTO PREDICTIONS VALUES (?, ?, ?, ?);"""
	cursor.execute(insertQuery, (time, text, prediction, prediction_time))

	connection.commit()
	cursor.close()
	connection.close()


@app.route('/predict', methods=['POST'])
def get_text():

	model = app.config['model']

	if request.method == "POST":

		data = request.get_json()

		if 'text' in data:
			currentDateTime = datetime.now()
			
			processor = ProcessText(data['text'])
			tokenize_text, pos_tags = processor.tokenizer()

			predictions = model_predict(tokenize_text, pos_tags, model)
			predictionDateTime = datetime.now()
			
			predictions_str = ""
			for token in predictions:
				predictions_str += token +  " "

			duration_seconds = (predictionDateTime - currentDateTime).total_seconds()

			save_user_imputs(currentDateTime, data['text'], predictions_str, duration_seconds)

			return jsonify({
				"ner_tags": predictions_str,
				"original": data['text'],
				"prediction_time": duration_seconds,
				"success": True,
				}), 200
		else:

			return jsonify({
				"success": False, 
				"error": "Something went wrong"
				}), 400

def create_app():
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)

	model_param = os.getenv('MODEL', 'biobert')
	logger.info(f"Model parameter: {model_param}")
	
	model = load_model(model_param)
	app.config['model'] = model

	# Perform a warm-up request
	warmup_text = "Warm-up text"
	logger.info("Performing warm-up request")
	processor = ProcessText(warmup_text)
	tokenize_text, pos_tags = processor.tokenizer()
	model_predict(tokenize_text, pos_tags, model)
	logger.info("Warm-up request completed")

	return app


app = create_app()
