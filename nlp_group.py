from datetime import datetime
from flask import Flask, jsonify, request
from flask_json import FlaskJSON, JsonError, json_response, as_json
from flask_restful import Resource, Api
import sqlite3
from process import *
from model_predict import *

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
	def get(self):
		return "Hello NLP!"

api.add_resource(HelloWorld, '/')


def save_user_imputs(time, text, prediction):

	connection = sqlite3.connect("user_input.db")
	cursor = connection.cursor()
	createTable = ''' CREATE TABLE IF NOT EXISTS PREDICTIONS (
		QueryDate TIMESTAMP,
		QueryText VARCHAR(400),
		Prediction VARCHAR(500)
		); '''
	cursor.execute(createTable)

	insertQuery = """INSERT INTO PREDICTIONS VALUES (?, ?, ?);"""
	cursor.execute(insertQuery, (time, text, prediction))

	connection.commit()
	cursor.close()
	connection.close()


@app.route('/prediction', methods=['POST'])
def get_text():
	if request.method == "POST":

		data = request.get_json()

		if 'text' in data:
			currentDateTime = datetime.now()
			
			processor = ProcessText(data['text'])
			tokenize_text, pos_tags = processor.tokenizer()

			predictions = model_predict(tokenize_text, pos_tags)
			
			predictions_str = ""
			for token in predictions:
				predictions_str += token +  " "

			save_user_imputs(currentDateTime, data['text'], predictions_str)

			return jsonify({
				"success": True,
				"original": data['text'],
				"ner_tags": predictions_str

				}), 200
		else:

			return jsonify({
				"success": False, 
				"error": "Something went wrong"
				}), 400


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)