
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from collections import Counter
from wordcloud import WordCloud
import seaborn as sns
import re



## EXTRACTING AND ANALYZING NER DATA FROM CONLL FILES

def read_conll_file(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:  
        sentence = []
        for line in file: 
            line = line.strip() 
            if line:  
                columns = line.split('\t') 
                sentence.append(columns)
            else:  
                if sentence: 
                    data.append(sentence) 
                    sentence = []  
        if sentence:
            data.append(sentence)
    return data

# Read the conll files
train_data = read_conll_file('conll2003.train.conll')
dev_data = read_conll_file('conll2003.dev.conll')
test_data = read_conll_file('conll2003.test.conll')


'''
The initial two sentences to get an overview of the appearance of our datasets.
Each sentence is a list of lists, where each inner list represents a word/token. 
Empty lines separate sentences.

'''

print("TRAINING DATASET")
for sentence in train_data[:2]:
    print('\n'.join('\t'.join(columns) for columns in sentence))
    print()  # Empty line to separate sentences
        
print(f"\nThe type of the training data is:", type(train_data)) 
print("-----------------------------------------------")

print(f"\nDEVELOPMENT DATASET")       
for sentence in dev_data[:2]:
    print('\n'.join('\t'.join(columns) for columns in sentence))
    print()  # Empty line to separate sentences
        
print(f"\nThe type of the development data is:", type(dev_data)) 
print("-----------------------------------------------")

print(f"\nTESTING DATASET")
for sentence in test_data[:2]:
    print('\n'.join('\t'.join(columns) for columns in sentence))
    print()  # Empty line to separate sentences
        
print(f"\nThe type of the testing data is:", type(test_data)) 



## DATA DISTRIBUTION 

'''
The number of instances we have per class.
What class is best and least represented.

'''

def count_ner_labels(data):
    ner_counts = {}   # A dictionary to store the counts of NER labels

    # Variables to keep track of the best represented and least represented classes
    best_represented_class = None
    least_represented_class = None
    max_count = 0
    min_count = float('inf')

    for sentence in data:
        for columns in sentence:
            ner_label = columns[3]  
            
            if ner_label not in ner_counts:
                ner_counts[ner_label] = 1
            else:
                ner_counts[ner_label] += 1

            # Check if the current class is the best represented
            if ner_counts[ner_label] > max_count:
                max_count = ner_counts[ner_label]
                best_represented_class = ner_label

    # Find the least represented class after all counts have been calculated
    for ner_label, count in ner_counts.items():
        if count < min_count:
            min_count = count
            least_represented_class = ner_label

    # The distribution of NER labels
    for label, count in ner_counts.items():
        print(f"The NER label '{label}' has {count} instances.")

    # The best and least represented classes
    print(f"\nThe best represented class is '{best_represented_class}', which has {max_count} instances.")
    print(f"The least represented class is '{least_represented_class}', which has {min_count} instances.")
    

    # Plot the distributions
    plt.figure(figsize=(10, 5))
    plt.bar(ner_counts.keys(), ner_counts.values(), log=True)  # Set log=True for a logarithmic scale
    plt.title('Distribution of NER Labels')
    plt.xlabel('NER Labels')
    plt.ylabel('Count (Log Scale)')
    plt.xticks(rotation=90)
    plt.show()
    
    return ner_counts, best_represented_class, least_represented_class


print(f"\n-----------------------------------------------")
print("DATA DISTRIBUTION")
print(f"\nTRAINING DATASET")
ner_counts, best_represented_class, least_represented_class = count_ner_labels(train_data)
print(f"\n-----------------------------------------------")
print(f"\n DEVELOPMENT DATASET")
ner_counts, best_represented_class, least_represented_class = count_ner_labels(dev_data)
print(f"\n-----------------------------------------------")
print(f"\n TESTING DATASET")
ner_counts, best_represented_class, least_represented_class = count_ner_labels(test_data)



## FEATURE EXPLORATION

'''
Hypotheses about linguistic (or orthographic) features.
Testing them by exploring their distribution in the data. 

'''

print(f"\n-----------------------------------------------")
print("Hypothesis 1: Words starting with uppercase letters are more likely to be named entities.")

# Identify capitalized words
capitalized_pattern = r'^[A-Z][a-z]*$|^([A-Z]+)$'

for sentence in train_data:
    for columns in sentence:
        word = columns[0]  # The word/token
        
        if re.match(capitalized_pattern, word):
            print(f"Word: '{word}' is capitalized.")
        else:
            print(f"Word: '{word}' is not capitalized.")
                
capitalized_entities = defaultdict(int)
total_entities = 0

for sentence in train_data:
    for columns in sentence:
        word = columns[0]  
        ner_label = columns[3]  
        
        # Exclude 'O' category
        if ner_label == 'O':
            continue
        
        # Hypothesis 1: Check if words that begin with capital letters are tagged as entities
        if word[0].isupper():
            capitalized_entities[ner_label] += 1
            total_entities += 1

for label, count in capitalized_entities.items():
    percentage = (count / total_entities) * 100
    print(f"NER label '{label}' has {percentage:.2f}% of entities starting with capital letters")


# Calculate the percentages
percentages = [(count / total_entities) * 100 for count in capitalized_entities.values()]

# Plot the distributions
plt.figure(figsize=(10, 6))
plt.bar(capitalized_entities.keys(), percentages, color='orange')
plt.xlabel('NER Labels')
plt.ylabel('Percentage')
plt.title('Percentage of Entities Starting with Capital Letters for Each NER Label')
plt.show()



print(f"\n-----------------------------------------------")
print("Hypothesis 2: Named entities may have unique word length traits, with some types having longer or shorter averages than non-entities.")

# Initialize counters for each named entity category
ner_category_counters = {ner_label: Counter() for ner_label in ner_counts.keys() if ner_label != 'O'}
ner_category_lengths = {ner_label: [] for ner_label in ner_counts.keys() if ner_label != 'O'}

for sentence in train_data:
    for columns in sentence:
        word = columns[0]  
        ner_label = columns[3]  

        # Exclude 'O' category
        if ner_label == 'O':
            continue
        
        ner_category_counters[ner_label][word] += 1
        ner_category_lengths[ner_label].append(len(word))

# Calculate average word length for each category
ner_category_avg_lengths = {ner_label: sum(lengths)/len(lengths) if lengths else 0 for ner_label, lengths in ner_category_lengths.items()}

for ner_label, avg_length in ner_category_avg_lengths.items():
    print(f"Average word length in category '{ner_label}': {avg_length}")
    
# Calculate the average word lengths
avg_lengths = list(ner_category_avg_lengths.values())

# Plot the distributions
plt.figure(figsize=(10, 6))
plt.bar(ner_category_avg_lengths.keys(), avg_lengths, color='orange')
plt.xlabel('NER Labels')
plt.ylabel('Average Word Length')
plt.title('Average Word Length for Each NER Category')
plt.show()
 
    
    
print(f"\n-----------------------------------------------")
print("Hypothesis 3: Certain words are more frequent in specific classes.")

# Initialize counters for each named entity category
ner_category_counters = {ner_label: Counter() for ner_label in ner_counts.keys() if ner_label != 'O'}

for sentence in train_data:
    for columns in sentence:
        word = columns[0]  
        ner_label = columns[3]  

        # Exclude 'O' category
        if ner_label == 'O':
            continue
        
        ner_category_counters[ner_label][word] += 1


for ner_label, word_counter in ner_category_counters.items():
    most_common_words = word_counter.most_common(5)  # The top 5 most common words
    print(f"Most common words in category '{ner_label}':")
    for word, count in most_common_words:
        print(f"  Word: '{word}', Count: {count}")
    print()

# Visualize the most common words in each category
for ner_label, word_counter in ner_category_counters.items():
    wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_counter)

    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.title(f"Most Common Words in Category '{ner_label}'")
    plt.axis("off")
    plt.show()
    
    
    
print(f"\n-----------------------------------------------")
print("Hypothesis 4: The NER label of a token is influenced by the NER label of the previous token.")

def analyze_previous_token(data):
    # Initialize a dictionary to store the counts
    ner_label_followers = {ner_label: {} for ner_label in set(columns[3] for sentence in data for columns in sentence)}

    for sentence in data:
        for i in range(1, len(sentence)):
            current_ner_label = sentence[i][3]
            previous_ner_label = sentence[i-1][3]

            if current_ner_label not in ner_label_followers[previous_ner_label]:
                ner_label_followers[previous_ner_label][current_ner_label] = 1
            else:
                ner_label_followers[previous_ner_label][current_ner_label] += 1

    return ner_label_followers

# Use the function to analyze the training data
ner_label_followers = analyze_previous_token(train_data)

# Print the results
for ner_label, followers in ner_label_followers.items():
    print(f"NER label '{ner_label}' is followed by:")
    for follower, count in followers.items():
        print(f"  - NER label '{follower}' {count} times")
        

# Convert the results to a DataFrame
df = pd.DataFrame(ner_label_followers).fillna(0)

# Plot the heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(df, cmap='viridis')
plt.title('Heatmap of NER Labels following each NER Label')
plt.xlabel('Current NER Label')
plt.ylabel('Previous NER Label')
plt.show()
        
