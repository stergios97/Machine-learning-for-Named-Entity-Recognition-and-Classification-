
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam

from data_extraction import train_data_subset_scaled, train_labels_subset, test_data_subset_scaled, test_labels_subset
from evaluation import evaluate_model




print("STARTING TRAINING OF MODELS...")

## TRADITIONAL ML MODELS

lr = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=1000) # lr with mix of word ebmeddings and one-hot encoded vectors, and the optimal hyperparameters
svm = SVC()   # svm with mix of word ebmeddings and one-hot encoded vectors
nb = GaussianNB()  # nb with mix of word ebmeddings and one-hot encoded vectors

# Train the models
lr.fit(train_data_subset_scaled, train_labels_subset)
svm.fit(train_data_subset_scaled, train_labels_subset)
nb.fit(train_data_subset_scaled, train_labels_subset)

# Make predictions on the testing data
lr_predictions = lr.predict(test_data_subset_scaled)
svm_predictions = svm.predict(test_data_subset_scaled)
nb_predictions = nb.predict(test_data_subset_scaled)



## LSTM with mix of word ebmeddings and one-hot encoded vectors

# Encode the labels
le = LabelEncoder()
train_labels_subset = le.fit_transform(train_labels_subset)
test_labels_lstm = le.transform(test_labels_subset)
train_labels_subset = np.array(train_labels_subset).astype(float)

# Optimizer and architecture
opt = Adam(learning_rate=0.001, clipvalue=0.5)
lstm = Sequential()
lstm.add(LSTM(100, activation='softmax', input_shape=(train_data_subset_scaled.shape[1], 1)))
lstm.add(Dense(1))
lstm.compile(optimizer=opt, loss='mse')

# Train the model
lstm.fit(train_data_subset_scaled, train_labels_subset, epochs=10, verbose=1)

# Make predictions on the testing data
test_data_subset_scaled = test_data_subset_scaled.reshape((test_data_subset_scaled.shape[0], test_data_subset_scaled.shape[1], 1))
lstm_predictions = lstm.predict(test_data_subset_scaled)

# Round predictions to the nearest integer and cast to int
lstm_predictions = np.round(lstm_predictions).astype(int)



# Print the evaluation scores
print('Logistic Regression:')
evaluate_model(lr_predictions, test_labels_subset)
print('\nSVM:')
evaluate_model(svm_predictions, test_labels_subset)
print('\n Gaussian Naive Bayes:')
evaluate_model(nb_predictions, test_labels_subset)
print('\nLSTM:')
evaluate_model(lstm_predictions, test_labels_lstm)
