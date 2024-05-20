import pytest
import torch
from transformers import BertTokenizerFast
from model_predict import BioBertPosTagsClassifier, model_predict

# Constants for tests
MODEL_PATH = 'models/biobert_postags_cased__e_16.pt'
CHECKPOINT = 'dmis-lab/biobert-v1.1'
OUTPUT_DIM = 4
POS_VOCAB_SIZE = 18
POS_EMBEDDING_DIM = 16

@pytest.fixture(scope="module")
def model():
    """ Load the BioBertPosTagsClassifier for testing. """
    model = BioBertPosTagsClassifier(output_dim=OUTPUT_DIM, pos_vocab_size=POS_VOCAB_SIZE, pos_embedding_dim=POS_EMBEDDING_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

@pytest.fixture(scope="module")
def tokenizer():
    """ Load the BertTokenizerFast for testing. """
    return BertTokenizerFast.from_pretrained(CHECKPOINT)

def test_model_loading(model):
    """ Test that the model loads correctly and is in eval mode. """
    assert model.training == False, "Model should be in eval mode."

def test_tokenization(tokenizer):
    """ Test the tokenizer output types and lengths. """
    sample_text = ['Hello', 'world', 'this', 'is', 'a', 'test']
    tokens = tokenizer(sample_text, return_tensors='pt', padding=True, truncation=True)
    assert 'input_ids' in tokens, "Tokenizer should produce input ids."
    assert 'attention_mask' in tokens, "Tokenizer should produce attention masks."
    assert tokens['input_ids'].size(1) <= 512, "Input ids should not exceed max length."

def test_end_to_end(model, tokenizer):
    """ End-to-end test of tokenization through model prediction. """
    tokens = ['This', 'is', 'a', 'test']
    pos_tags = ['DET', 'VERB', 'DET', 'NOUN']
    predictions = model_predict(tokens, pos_tags)
    assert len(predictions) == len(tokens), "Should return a prediction for each token."
    assert all(isinstance(tag, str) for tag in predictions), "All predictions should be string labels."

def test_model_predict_values(model, tokenizer):
    """ Test the values of the model prediction. """
    tokens = ['For', 'this', 'purpose', 'the', 'Gothenburg', 'Young', 'Persons', 'Empowerment', 'Scale', '(', 'GYPES', ')', 'was', 'developed', '.']
    pos_tags = ['ADP', 'DET', 'NOUN', 'DET', 'PROPN', 'PROPN', 'PROPN', 'PROPN', 'PROPN', 'PUNCT', 'PROPN', 'PUNCT', 'AUX', 'VERB', 'PUNCT']
    true_ner_tags = ['B-O', 'B-O', 'B-O', 'B-O', 'B-LF', 'I-LF', 'I-LF', 'I-LF', 'I-LF', 'B-O', 'B-AC', 'B-O', 'B-O', 'B-O', 'B-O']
    predictions = model_predict(tokens, pos_tags)
    assert predictions == true_ner_tags, "Predictions should match the expected values."
