
from itertools import combinations
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

from data_extraction import train_data_subset_scaled, train_labels_subset, dev_data_subset_scaled, dev_labels_subset



# List of all features we will use
selected_features = ['token', 'POS-tag', 'chunk-tag', 'capitalized', 'word_length', 'prev_token']


# Gaussian Naive Bayes model
nb = GaussianNB()


# Variables to keep track of the best score and corresponding feature combination
best_score = 0
best_features = None

# Try all combinations of features
for r in range(1, len(selected_features) + 1):
    for features_subset in combinations(selected_features, r):
        # Select only the features in features_subset
        selected_features_indices = [selected_features.index(f) for f in features_subset]
        training_data = train_data_subset_scaled[:, selected_features_indices]
        develop_data = dev_data_subset_scaled[:, selected_features_indices]

        # Fit the model with the subset of features and calculate the score
        nb.fit(training_data, train_labels_subset)
        score = f1_score(dev_labels_subset, nb.predict(develop_data), average='macro')

        # If this score is better than the current best score, update the best score and features
        if score > best_score:
            best_score = score
            best_features = features_subset

        print(f"\nFEATURE ABLATION ANALYSIS")
        print(f"Using features {features_subset} gives a (Macro) F1 score of {score}")

print(f"\nThe best feature combination is {best_features}, which gives a (Macro) F1 score of {best_score}")
        