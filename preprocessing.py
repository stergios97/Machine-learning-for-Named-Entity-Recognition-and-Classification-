
import numpy as np


# Remove 'B-' and 'I-' from NER labels (flat encoding)
def main_label(label):
    '''
    Expressing entities without the B- (Beginner)
    and I- (Insider) prefixes, simplifying the labels
    to the direct entity type for each token.

    '''
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]
    return label



def flatten_features(data):
    '''
    Flatten my data to ensure it conforms to the appropriate structure for my models.

    '''
    return np.array([np.concatenate([np.atleast_1d(x) for x in row]) for row in data.values])
