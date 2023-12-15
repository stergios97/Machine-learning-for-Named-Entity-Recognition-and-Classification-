
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# Evaluate the models
def evaluate_model(predictions, labels):
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='macro')
    cm = confusion_matrix(labels, predictions)

    print(f'Precision: {precision}\nRecall: {recall}\nF1-score: {f1}\nConfusion Matrix:\n{cm}')

