

import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from features import check_capitalization, get_previous_token, one_hot_encoding_tags, word_embeddings, word_embedding_model
from preprocessing import main_label, flatten_features




# Feature extraction
def extract_features_and_labels(trainingfile):   
    data = []
    targets = []
    tokens = []  # List to store all tokens
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()  # Split by lines in the file
            if len(components) > 0:   # Skip the empty lines
                token = components[0]
                tokens.append(token)  
                capitalized = check_capitalization(token)
                pos_tag = components[1]
                chunk_tag = components[2]
                word_length = len(token)
                prev_token = get_previous_token(tokens, len(tokens) - 1)  
                feature_dict = {'token':token, 'POS-tag': pos_tag, 'chunk-tag': chunk_tag, 'capitalized': capitalized, 'word_length': word_length, 'prev_token': prev_token}
                data.append(feature_dict) 
                ner_label = components[-1]
                ner_label = main_label(ner_label)
                targets.append(ner_label)
    return data, targets


# All the files
train_conll_file = 'conll2003.train.conll' 
dev_conll_file = 'conll2003.dev.conll'  
test_conll_file = 'conll2003.test.conll'


# Concatenate the files into one
with open('data.conll', 'w') as outfile:
    for fname in [train_conll_file, dev_conll_file, test_conll_file]:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)

data, labels = extract_features_and_labels('data.conll')

# Print the first tokens of combined data
for i, (features, label) in enumerate(zip(data, labels)):
    print(f"Row {i + 1} of combined data:")
    print("Features:", features)
    print("Label:", label)
    print()
    if i == 2:  
        break



print(f"\nAPPLYING ONE-HOT ENCODING")
encoder = OneHotEncoder(sparse_output=False)
data = one_hot_encoding_tags(data, encoder)

# Print the first rows of combined data after applying one-hot encoding
for i, (features, label) in enumerate(zip(data, labels)):
    print(type(data))
    print(f"Row {i + 1}:")
    print("Features:", features)
    print("Label:", label)
    print()
    if i == 1:  
        break
    
    
print(f"\nAPPLYING WORD EMBEDDINGS")
# Use word-embeddings for the tokens and the previous tokens
data = word_embeddings(word_embeddings(data, word_embedding_model, 'token'), word_embedding_model, 'prev_token')

# Print the first rows of combined data after applying word embeddings 
for i, (features, label) in enumerate(zip(data, labels)):
    print(type(data))
    print(f"Row {i + 1}:")
    print("Features:", features)
    print("Label:", label)
    print()
    if i == 1:  
        break


print(f"\nFLATTENING DATA")
data_df = pd.DataFrame(data)
flattened_data = flatten_features(data_df)

print("Shape of flattened data:", flattened_data.shape)



# Split the combined data into their original separate datasets
train_len = sum(1 for line in open(train_conll_file))
dev_len = sum(1 for line in open(dev_conll_file))
test_len = sum(1 for line in open(test_conll_file))

train_data_flattened = flattened_data[:train_len]
dev_data_flattened = flattened_data[train_len:train_len+dev_len]
test_data_flattened = flattened_data[train_len+dev_len:train_len+dev_len+test_len]

# Split the labels back into separate datasets
train_labels = labels[:train_len]
dev_labels = labels[train_len:train_len+dev_len]
test_labels = labels[train_len+dev_len:train_len+dev_len+test_len]


# Use a subset of my data to train and evaluate my models (computational and memory capacity reasons)
# Calculate 50% of the data size
subset_size_train = int(len(train_data_flattened) * 0.5)
subset_size_dev = int(len(dev_data_flattened) * 0.5)
subset_size_test = int(len(test_data_flattened) * 0.5)

# Select a subset(50%) of the original data
train_data_subset = train_data_flattened[:subset_size_train]
dev_data_subset = dev_data_flattened[:subset_size_dev]
test_data_subset = test_data_flattened[:subset_size_test]

print("Shape of train_data_subset:", train_data_subset.shape)
print("Shape of dev_data_subset:", dev_data_subset.shape)
print("Shape of test_data_subset:", test_data_subset.shape)

# Select a subset(50%) of the labels
train_labels_subset = train_labels[:subset_size_train]
dev_labels_subset = dev_labels[:subset_size_dev]
test_labels_subset = test_labels[:subset_size_test]

# Scale the data
scaler = StandardScaler()
train_data_subset_scaled = scaler.fit_transform(train_data_subset)
dev_data_subset_scaled = scaler.transform(dev_data_subset)
test_data_subset_scaled = scaler.transform(test_data_subset)



