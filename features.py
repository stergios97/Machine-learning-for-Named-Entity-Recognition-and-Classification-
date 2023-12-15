
from gensim.models import KeyedVectors


# Capturing various forms of capitalization (all caps, first letter capitalized, and title case)
def check_capitalization(token):
    if token.isupper() or token[0].isupper() or token.istitle():  
        return 1
    else:
        return 0 
    
    
# Contextual features (previous token)
def get_previous_token(tokens, index):
    return tokens[index-1] if index > 0 else 'None'


# One-hot encoding for 'POS-tags', and 'Chunk-tags'
def one_hot_encoding_tags(data, encoder):
    pos_tags = [item['POS-tag'] for item in data]
    chunk_tags = [item['chunk-tag'] for item in data]
    
    pos_tags_encoded = encoder.fit_transform([[tag] for tag in pos_tags])
    chunk_tags_encoded = encoder.fit_transform([[tag] for tag in chunk_tags])
    
    for i, item in enumerate(data):
        item['POS-tag'] = pos_tags_encoded[i]
        item['chunk-tag'] = chunk_tags_encoded[i]
        
    return data


# Word embeddings (Word2Vec) for tokens and previous tokens
# Load the word embedding model
word_embedding_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def word_embeddings(data, word_embedding_model, token_type):
    for i, item in enumerate(data):
        token = item[token_type]
        
        # Check if the token exists in the word embedding model
        if token in word_embedding_model:
            data[i][token_type] = word_embedding_model[token]
        else:
            data[i][token_type] = [0]*300  # Replace with a zero vector if the token is not in the model
            
    return data




