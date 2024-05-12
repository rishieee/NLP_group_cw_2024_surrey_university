import torch
import torch.nn as nn
import numpy as np
from torchtext.vocab import vocab
from transformers import BertTokenizerFast, BertModel

class BioBertPosTagsClassifier(nn.Module):
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
        adjusted_pos_tags = torch.where(pos_tags == -100, torch.tensor(self.pos_embedding.padding_idx, device=pos_tags.device), pos_tags)

        # Get embeddings from POS tags
        pos_embeddings = self.pos_embedding(adjusted_pos_tags)

        # Concatenate BERT and POS embeddings
        combined_embeddings = torch.cat((sequence_output, pos_embeddings), dim=-1)

        logits = self.fc(combined_embeddings)

        return logits

def model_predict(tokenize_text, pos_tags, device='cpu'):
    """
    Make a prediction from tokenized text and POS tags using the given model.

    Args:
    tokenize_text (list of str): List of tokens.
    pos_tags (list of str): List of POS tags corresponding to each token.
    model (BioBertPosTagsClassifier): The trained model.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer used during model training.
    pos_tag_vocab (dict): Dictionary mapping POS tag strings to indices.
    device (str): Device to run the model ('cpu' or 'cuda').

    Returns:
    torch.Tensor: The logits returned by the model.
    """
    # Load model
    output_dim = 4
    pos_vocab_size = 18
    pos_embedding_dim = 16

    model = BioBertPosTagsClassifier(output_dim=output_dim, pos_vocab_size=pos_vocab_size, pos_embedding_dim=pos_embedding_dim)
    model.load_state_dict(torch.load('models/biobert_postags_cased__e_16.pt', map_location=torch.device('cpu')))

    model = model.to(device)
    model.eval()

    # Load tokenizer
    check_point = 'dmis-lab/biobert-v1.1'
    tokenizer = BertTokenizerFast.from_pretrained(check_point)

    # Define the NER tag vocabulary
    #ner_tag_to_ix = {'B-O': 0, 'B-AC': 1, 'B-LF': 2, 'I-LF': 3}
    ix_to_ner_tag = {0: 'B-O', 1: 'B-AC', 2: 'B-LF', 3: 'I-LF'}

    # Define the POS tag vocabulary
    PAD_TOKEN = '<pad>'
    pos_tag_to_ix = {'ADV': 0, 'VERB': 1, 'SCONJ': 2, 'PRON': 3, 'PROPN': 4, 'NOUN': 5, 'ADP': 6, 'ADJ': 7, 'CCONJ': 8, 'SYM': 9, 'NUM': 10, 'PART': 11, 'PUNCT': 12, 'DET': 13, 'INTJ': 14, 'AUX': 15, 'X': 16}
    pos_tag_vocab = vocab(pos_tag_to_ix, min_freq=0, specials=(PAD_TOKEN,), special_first=True)

    # Tokenize the input text
    encoded_tokens = tokenizer(tokenize_text,
                               is_split_into_words=True,
                               padding='max_length',
                               truncation=True,
                               max_length=512,
                               return_attention_mask=True,
                               return_tensors='pt')

    # Convert pos_tags using the vocabulary
    pos_tags_idx = [pos_tag_vocab[tag] for tag in pos_tags]

    encoded_pos_tags = np.ones(len(encoded_tokens['input_ids'][0]), dtype=int) * -100

    token_ids_word_mapping = encoded_tokens.word_ids(batch_index=0)

    prev_word_idx = None
    for i, word_idx in enumerate(token_ids_word_mapping):
        # Check if the token was in the orginal sequence (otherwise will keep the -100)
        if (word_idx is not None) and (word_idx != prev_word_idx):
            # Assign to each token (or subtoken) the same label associated to the token
            # in the original sentence
            encoded_pos_tags[i] = pos_tags_idx[word_idx]
        prev_word_idx = word_idx

    # Convert to tensor
    token_ids = encoded_tokens['input_ids'].to(device)
    attention_mask = encoded_tokens['attention_mask'].to(device)
    pos_tags_tensor = torch.tensor(encoded_pos_tags, dtype=torch.long).unsqueeze(0).to(device)

    # Forward pass through the model
    with torch.no_grad():
        logits = model(token_ids, attention_mask, pos_tags_tensor)

        # Get the predictions
        _, predicted_ner_tags = torch.max(logits, dim=2)

        # Update lists of true and predicted tags, excluding ignored index (-100 used for padding in BERT)
        active_accuracy = (pos_tags_tensor != -100).cpu()
        predicted_ner_tags = torch.masked_select(predicted_ner_tags.cpu(), active_accuracy).cpu().numpy()

    return [ix_to_ner_tag[p] for p in predicted_ner_tags]

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
