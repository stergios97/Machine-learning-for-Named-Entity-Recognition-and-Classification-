
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

from data_extraction import train_data_subset_scaled, train_labels_subset, test_data_subset_scaled, test_labels_subset



# Initialize the model in which we will do error analysis
nb = GaussianNB()

# Reshape the testing_data_subset back to 2 dimensions
test_data_subset_scaled_2d = test_data_subset_scaled.reshape(test_data_subset_scaled.shape[0], -1)

# Predict the test set results
nb_predictions = nb.predict(test_data_subset_scaled_2d)

# Print a classification report
print(classification_report(test_labels_subset, nb_predictions, zero_division=1))

# Compute the confusion matrix
cm = confusion_matrix(test_labels_subset, nb_predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix for Gaussian Naive Bayes')

# Feature correlation analysis
correlation_matrix = pd.DataFrame(train_data_subset_scaled).corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f')
plt.title('Feature Correlation Matrix for Gaussian Naive Bayes')
plt.show()