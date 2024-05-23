import torch
import torch.nn as nn
import numpy as np
from torchtext.vocab import vocab
from torch.quantization import quantize_dynamic
from transformers import BertTokenizerFast, BertModel
import nltk

nltk.download('punkt')

# Constants
MODEL_CHECKPOINT = 'dmis-lab/biobert-v1.1'
MODEL_PATH = 'models/biobert_postags_cased__e_16.pt'
MAX_LENGTH = 512
PAD_TOKEN = '<pad>'

# Define POS and NER tags vocabulary
POS_TAGS = {
    'ADV': 0, 'VERB': 1, 'SCONJ': 2, 'PRON': 3, 'PROPN': 4, 'NOUN': 5, 'ADP': 6,
    'ADJ': 7, 'CCONJ': 8, 'SYM': 9, 'NUM': 10, 'PART': 11, 'PUNCT': 12, 'DET': 13,
    'INTJ': 14, 'AUX': 15, 'X': 16
}
NER_TAGS = {0: 'B-O', 1: 'B-AC', 2: 'B-LF', 3: 'I-LF'}

class BioBertPosTagsClassifier(nn.Module):
    """BERT model with POS tags classifier."""
    def __init__(self, output_dim, pos_vocab_size, pos_embedding_dim):
        super(BioBertPosTagsClassifier, self).__init__()
        self.model_label = 'biobert_postags_cased'
        self.bert = BertModel.from_pretrained('dmis-lab/biobert-v1.1',
                                              num_labels=output_dim,
                                              add_pooling_layer=False)

        # Add 1 to pos_vocab_size to account for the special -100 index.
        # We'll reserve the last embedding vector for -100 indices.
        self.pos_embedding = nn.Embedding(num_embeddings=pos_vocab_size + 1,
                                          embedding_dim=pos_embedding_dim,
                                          padding_idx=pos_vocab_size)

        # Adjust the input size of the classifier
        combined_embedding_dim = self.bert.config.hidden_size + pos_embedding_dim
        self.fc = nn.Linear(combined_embedding_dim, output_dim)

    def forward(self, text, attention_mask, pos_tags):
        outputs = self.bert(text, attention_mask=attention_mask, return_dict=False)

        sequence_output = outputs[0]    # [batch_size, sequence_length, 768]

        # Adjust pos_tags to ensure -100 indices map to the last embedding vector
        adjusted_pos_tags = torch.where(pos_tags == -100,
                                        torch.tensor(self.pos_embedding.padding_idx, device=pos_tags.device),
                                        pos_tags)

        # Get embeddings from POS tags
        pos_embeddings = self.pos_embedding(adjusted_pos_tags)

        # Concatenate BERT and POS embeddings
        combined_embeddings = torch.cat((sequence_output, pos_embeddings), dim=-1)

        logits = self.fc(combined_embeddings)

        return logits

def quantise_model(model):
    # Set the quantization engine
    torch.backends.quantized.engine = 'qnnpack'

    # Apply dynamic quantization
    quantised_model = quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return quantised_model

def load_model(model_name, device='cpu'):
    """Load model."""
    model = BioBertPosTagsClassifier(output_dim=len(NER_TAGS), pos_vocab_size=len(POS_TAGS)+1, pos_embedding_dim=16)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))
    
    if model_name == "quantized":
        model = quantise_model(model)

    model.to(device).eval()
    return model


def load_tokenizer():
    """Load tokenizer."""
    return BertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)


def encode_tokens(text):
    """Encode tokens to tensor."""
    tokenizer = load_tokenizer()
    return tokenizer(text,
                     is_split_into_words=True,
                     padding='max_length',
                     truncation=True,
                     max_length=MAX_LENGTH,
                     return_attention_mask=True,
                     return_tensors='pt')


def encode_pos_tags(pos_tags, encoded_tokens):
    """Encode POS tags to tensor."""
    pos_tag_vocab = vocab(POS_TAGS, min_freq=0, specials=(PAD_TOKEN,), special_first=True)

    pos_tags_idx = [pos_tag_vocab[tag] for tag in pos_tags]

    encoded_pos_tags = np.ones(len(encoded_tokens['input_ids'][0]), dtype=int) * -100

    token_ids_word_mapping = encoded_tokens.word_ids(batch_index=0)

    prev_word_idx = None
    for i, word_idx in enumerate(token_ids_word_mapping):
        # Check if the token was in the original sequence (otherwise will keep the -100)
        if (word_idx is not None) and (word_idx != prev_word_idx):
            # Assign to each token (or sub token) the same label associated to the token
            # in the original sentence
            encoded_pos_tags[i] = pos_tags_idx[word_idx]
        prev_word_idx = word_idx

    return torch.tensor(encoded_pos_tags, dtype=torch.long).unsqueeze(0)


def logits_to_ner_tags(logits, encoded_pos_tags_tensor):
    """Convert model logits to NER tags."""
    _, predicted_ner_tags = torch.max(logits, dim=2)
    active_accuracy = (encoded_pos_tags_tensor != -100).cpu()
    predicted_ner_tags = torch.masked_select(predicted_ner_tags.cpu(), active_accuracy).cpu().numpy()

    return [NER_TAGS[p] for p in predicted_ner_tags]

def model_predict(tokenize_text, pos_tags, model, device='cpu'):
    """Predict NER tags for a given text."""
    # Load model
    #model = load_model(device)

    # Encode tokens
    encoded_tokens = encode_tokens(tokenize_text)

    # Encode POS tags
    encoded_pos_tags = encode_pos_tags(pos_tags, encoded_tokens)

    # Convert to tensor
    encoded_tokens_tensor = encoded_tokens['input_ids'].to(device)
    attention_mask_tensor = encoded_tokens['attention_mask'].to(device)
    encoded_pos_tags_tensor = encoded_pos_tags.to(device)

    # Forward pass through the model
    with torch.no_grad():
        logits = model(encoded_tokens_tensor, attention_mask_tensor, encoded_pos_tags_tensor)

    return logits_to_ner_tags(logits, encoded_pos_tags_tensor)

# Example usage
# Assume model, tokenizer, and pos_tag_vocab are already defined and loaded
#tokens = ['For', 'this', 'purpose', 'the', 'Gothenburg', 'Young', 'Persons', 'Empowerment', 'Scale', '(', 'GYPES', ')', 'was', 'developed', '.']
#pos_tags = ['ADP', 'DET', 'NOUN', 'DET', 'PROPN', 'PROPN', 'PROPN', 'PROPN', 'PROPN', 'PUNCT', 'PROPN', 'PUNCT', 'AUX', 'VERB', 'PUNCT']
#true_ner_tags = ['B-O', 'B-O', 'B-O', 'B-O', 'B-LF', 'I-LF', 'I-LF', 'I-LF', 'I-LF', 'B-O', 'B-AC', 'B-O', 'B-O', 'B-O', 'B-O']
#predicted_ner_tags = model_predict(tokenize_text=tokens, pos_tags=pos_tags)
#print(tokens)
#print(pos_tags)
#print(predicted_ner_tags)
#print(true_ner_tags)
