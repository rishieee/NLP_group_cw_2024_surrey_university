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


if __name__ == '__main__':
	model_param = "biobert"
	if len(sys.argv) > 1:
		model_param = sys.argv[1]
	model = load_model(model_param)
	app.config['model'] = model
	app.run(host='0.0.0.0', port=5000)
