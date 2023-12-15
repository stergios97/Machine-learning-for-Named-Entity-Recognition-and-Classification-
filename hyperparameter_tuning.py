
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from data_extraction import train_data_subset_scaled, train_labels_subset, dev_data_subset_scaled


# Hyperparameters 
param_distributions = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['lbfgs', 'liblinear', 'sag', 'saga']
}

# Initialize the Logistic Regression model 
lr = LogisticRegression(max_iter=1000)

# RandomizedSearchCV
random_search = RandomizedSearchCV(lr, param_distributions, n_iter=5, cv=5, verbose=3)
random_search.fit(train_data_subset_scaled, train_labels_subset)

# Print the best parameters and the best score
print(f"\n HYPERPARAMETER TUNING")
print("The best hyperparameters are:", random_search.best_params_)
print("The best score for these:", random_search.best_score_)

# Predict on the development data using the best model
log_reg_best = random_search.best_estimator_
log_reg_predictions = log_reg_best.predict(dev_data_subset_scaled)