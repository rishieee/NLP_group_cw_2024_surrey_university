from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer
from model import BioBertPosTagsClassifier
from dataset import GenericBERTAbbreviationDataset

app = Flask(__name__)

# Load your tokenizer
tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1')

# Assume pos_vocab_size and pos_embedding_dim are known
output_dim = 4  # This should be the actual number of NER tags in your dataset
pos_vocab_size = 18  # This should be the actual size of your POS vocabulary
pos_embedding_dim = 16  # This should be your actual embedding dimension size

# Load your model
model = BioBertPosTagsClassifier(output_dim=output_dim, pos_vocab_size=pos_vocab_size, pos_embedding_dim=pos_embedding_dim)
model.load_state_dict(torch.load('models/biobert_postags_cased__e_16.pt', map_location=torch.device('cpu')))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input data
    data = request.json
    tokens = data['tokens']
    pos_tags = data['pos_tags']
    ner_tags = data['ner_tags']

    # Prepare data for the model
    dataset = GenericBERTAbbreviationDataset(tokenizer, [tokens], [pos_tags], [ner_tags], pos_tag_vocab, ner_tag_vocab)
    sample = dataset[0]  # Since we are only sending one example at a time

    with torch.no_grad():
        logits = model(sample['token_ids'].unsqueeze(0),  # Add batch dimension
                       sample['attention_mask'].unsqueeze(0),  # Add batch dimension
                       sample['pos_tags'].unsqueeze(0))  # Add batch dimension
        predictions = torch.argmax(logits, dim=-1)

    # Convert predictions to labels
    predicted_labels = [dataset.ner_tag_vocab.inverse[p] for p in predictions[0].numpy()]

    return jsonify({'predictions': predicted_labels})

if __name__ == '__main__':
    app.run(debug=True)
