from web_app.model_predict import model_predict, load_model, load_tokenizer, encode_tokens

from transformers import BertTokenizerFast

MODEL_CHECKPOINT = 'dmis-lab/biobert-v1.1'

POS_TAGS = {
    'ADV': 0, 'VERB': 1, 'SCONJ': 2, 'PRON': 3, 'PROPN': 4, 'NOUN': 5, 'ADP': 6,
    'ADJ': 7, 'CCONJ': 8, 'SYM': 9, 'NUM': 10, 'PART': 11, 'PUNCT': 12, 'DET': 13,
    'INTJ': 14, 'AUX': 15, 'X': 16
}
NER_TAGS = {0: 'B-O', 1: 'B-AC', 2: 'B-LF', 3: 'I-LF'}

def test_load_model_quantized():
    """ Test that the model loads correctly and is in eval mode. """
    model = load_model('quantized')
    assert model.training == False, "Model should be in eval mode."

def test_load_model_biobert():
    """ Test that the model loads correctly and is in eval mode. """
    model = load_model('biobert')
    assert model.training == False, "Model should be in eval mode."

def test_load_tokenizer():
    """Test that the tokenizer loads correctly and is an instance of BertTokenizerFast."""
    tokenizer = load_tokenizer()

    assert tokenizer is not None, "Tokenizer should not be None"
    assert isinstance(tokenizer, BertTokenizerFast), "Tokenizer should be an instance of BertTokenizerFast"
    assert tokenizer.name_or_path == MODEL_CHECKPOINT, f"Tokenizer should be loaded from {MODEL_CHECKPOINT}"

def test_encode_tokens():
    """ Test the tokenizer output types and lengths. """
    sample_text = ['Hello', 'world', 'this', 'is', 'a', 'test']
    encoded_tokens = encode_tokens(sample_text)
    assert 'input_ids' in encoded_tokens, "Tokenizer should produce input ids."
    assert 'attention_mask' in encoded_tokens, "Tokenizer should produce attention masks."
    assert encoded_tokens['input_ids'].size(1) <= 512, "Input ids should not exceed max length."

def test_model_predict_quantised():
    """ Test the values of the model prediction. """
    tokens = ['For', 'this', 'purpose', 'the', 'Gothenburg', 'Young', 'Persons', 'Empowerment', 'Scale', '(', 'GYPES', ')', 'was', 'developed', '.']
    pos_tags = ['ADP', 'DET', 'NOUN', 'DET', 'PROPN', 'PROPN', 'PROPN', 'PROPN', 'PROPN', 'PUNCT', 'PROPN', 'PUNCT', 'AUX', 'VERB', 'PUNCT']
    true_ner_tags = ['B-O', 'B-O', 'B-O', 'B-O', 'B-LF', 'I-LF', 'I-LF', 'I-LF', 'I-LF', 'B-O', 'B-AC', 'B-O', 'B-O', 'B-O', 'B-O']
    model = load_model('quantized')
    predictions = model_predict(tokens, pos_tags, model)
    assert len(predictions) == len(tokens), "Should return a prediction for each token."
    assert all(isinstance(tag, str) for tag in predictions), "All predictions should be string labels."
    assert predictions == true_ner_tags, "Predictions should match the expected values."

def test_model_predict_biobert():
    """ Test the values of the model prediction. """
    tokens = ['For', 'this', 'purpose', 'the', 'Gothenburg', 'Young', 'Persons', 'Empowerment', 'Scale', '(', 'GYPES', ')', 'was', 'developed', '.']
    pos_tags = ['ADP', 'DET', 'NOUN', 'DET', 'PROPN', 'PROPN', 'PROPN', 'PROPN', 'PROPN', 'PUNCT', 'PROPN', 'PUNCT', 'AUX', 'VERB', 'PUNCT']
    true_ner_tags = ['B-O', 'B-O', 'B-O', 'B-O', 'B-LF', 'I-LF', 'I-LF', 'I-LF', 'I-LF', 'B-O', 'B-AC', 'B-O', 'B-O', 'B-O', 'B-O']
    model = load_model('biobert')
    predictions = model_predict(tokens, pos_tags, model)
    print(predictions)
    assert len(predictions) == len(tokens), "Should return a prediction for each token."
    assert all(isinstance(tag, str) for tag in predictions), "All predictions should be string labels."
    assert predictions == true_ner_tags, "Predictions should match the expected values."