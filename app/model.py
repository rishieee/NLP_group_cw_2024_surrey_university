import torch.nn as nn
from transformers import BertConfig, BertModel

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
