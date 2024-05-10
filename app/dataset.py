from torch.utils.data import Dataset

class GenericBERTAbbreviationDataset(Dataset):
    def __init__(self, tokenizer, tokens, pos_tags, ner_tags, pos_tag_vocab, ner_tag_vocab, max_length=512):
        self.tokens = tokens
        self.pos_tags = pos_tags
        self.ner_tags = ner_tags
        self.pos_tag_vocab = pos_tag_vocab
        self.ner_tag_vocab = ner_tag_vocab
        self.tokenizer = tokenizer
        self.max_length = max_length

    def get_original_tokens_at_index(self, idx):
        return self.tokens[idx]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        ner_tags = [self.ner_tag_vocab[tag] for tag in self.ner_tags[idx]]
        pos_tags = [self.pos_tag_vocab[tag] for tag in self.pos_tags[idx]]

        # Tokenize and pad the input up to max_length
        encoded_tokens = self.tokenizer(tokens,
                                                                        is_split_into_words=True,
                                                                        max_length=self.max_length,
                                                                        padding='max_length',
                                                                        truncation=True,
                                                                        return_attention_mask=True,
                                                                        return_tensors='pt')

        # Make sure that the encoded_ner_tag sequence is aligned with the encoded_token sequence
        # Initialise an array of all -100s
        encoded_ner_tags = np.ones(len(encoded_tokens['input_ids'][0]), dtype=int) * -100
        encoded_pos_tags = np.ones(len(encoded_tokens['input_ids'][0]), dtype=int) * -100

        # gather the mapping between the original and the encoded token sequences
        # The encoded_tokens.word_ids(batch_index=0) call generates a
        # mapping from token indices to word indices in the original sentence.
        # Example Mapping: [None, 0, 0, 1, 2, 2, 3, None]

        token_ids_word_mapping = encoded_tokens.word_ids(batch_index=0)
        prev_word_idx = None
        for i, word_idx in enumerate(token_ids_word_mapping):
            # Check if the token was in the orginal sequence (otherwise will keep the -100)
            if (word_idx is not None) and (word_idx != prev_word_idx):
                # Assign to each token (or subtoken) the same label associated to the token
                # in the original sentence
                encoded_ner_tags[i] = ner_tags[word_idx]
                encoded_pos_tags[i] = pos_tags[word_idx]
            prev_word_idx = word_idx

        return {
            'token_ids': encoded_tokens['input_ids'].squeeze(),
            'attention_mask': encoded_tokens['attention_mask'].squeeze(),
            'ner_tags': torch.tensor(encoded_ner_tags, dtype=torch.long),
            'pos_tags': torch.tensor(encoded_pos_tags, dtype=torch.long),
            'idx': idx    # Return the index of the item
        }
