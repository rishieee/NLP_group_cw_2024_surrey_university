import numpy as np
import torch
import nltk
import pandas as pd
import os
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import spacy

class ProcessText:

	def __init__(self, text):
		self.text = text


	def tokenizer(self):
		text_tokenize = word_tokenize(self.text)
		#pos_tags = pos_tag(text_tokenize, tagset='universal')

		pos_tags_list = []
		nlp=spacy.load('en_core_web_sm')

		for token in nlp(self.text): 

			pos_tags_list.append(token.pos_)

		return text_tokenize, pos_tags_list



# data = "For this purpose the Gothenburg Young Persons Empowerment Scale (GYPES) was developed."
# processor = ProcessText(data)
# tokenize_text, pos_tags = processor.tokenizer()

# print(f"{tokenize_text} \n {pos_tags}")
