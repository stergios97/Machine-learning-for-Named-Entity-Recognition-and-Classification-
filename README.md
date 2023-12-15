# Machine learning for Named Entity Recognition and Classification (A project in the context of "Machine Learning for NLP" course)

## Overview

This file provides tools for extracting, analyzing, and exploring Named Entity Recognition (NER) data from CoNLL files. The code covers aspects such as Exploratory Data Analysis (EDA), Data Extraction, Data Preprocessing, Feature Exploration,  Model Evaluation, Hyperparameter Tuning, and Feature Ablation.
 The README below will guide you through the code and provide instructions for using it.

## Dependencies

Make sure you have the required Python libraries installed. You can install them using the following command:

```bash
pip install matplotlib pandas gensim scikit-learn numpy keras seaborn wordcloud
```

Additionally, make sure you have the Word2Vec model file "GoogleNews-vectors-negative300.bin" in the same directory as the scripts. You can download it from [Google's Word2Vec](https://code.google.com/archive/p/word2vec/)

## Navigation Path

1) Exploratory Data Analysis:
   - Run the 'eda.py' script to analyze the data, including NER label counts, identification of best and least represented classes, and visualization of the distribution through a bar chart. Additionally, test hypotheses about linguistic or orthographic features by examining their distribution in the dataset.

2) Data Preprocessing:
   - Run the 'preprocessing.py' script to preprocess the NER labels, and prepare the data for further analysis.
  
3) Features:
   - Run the 'features.py' script to implement the features that will be utilized for subsequent analysis. Additionally, the 'word_embedding_model' is present in this script.

4) Data Extraction:
   - Run the 'data_extraction.py' script to extract data (features and labels) from the training, development, and testing datasets. Additionally, the script will preprocess the data into a format suitable for input into the models.
  
5) Evaluation:
   - Run the 'evaluation.py' script to prepare it for evaluating the models in the last steps.

6) Hyperparameter-Tuning:
   - Run the 'hyperparameter_tuning.py' script to identify the optimal hyperparameters for the Logistic Regression model.

7) Feature Ablation:
   - Run the 'feature_ablation.py' script to determine the most effective combinations of features for the Gaussian Naive Bayes model.

8) Evaluate the models:
   - Run the 'models.py' script to train the SVM, Logistic Regression, Gaussian Naive Bayes, and LSTM network models, and obtain the evaluation results.

9) Error Analysis:
   - Run the 'error_analysis.py' script to do an error analysis for the Gaussian Naive Bayes model.

## How to Use

1) Ensure you have the required dependencies installed.
2) Download the Word2Vec model file GoogleNews-vectors-negative300.bin.
3) Place the Word2Vec model file in the same directory as the scripts.
4) Follow the navigation path to use specific functions or sections of the code.
5) Execute the code sections in your Python environment (preferably using Visual Studio Code).
