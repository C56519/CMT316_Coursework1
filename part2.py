import re
import numpy as np
import nltk
import operator
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from gensim.models import Word2Vec
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('vader_lexicon')


# 1 functions
def load_data():
    """
    Function: Load all news datasets
    :return: full_dataset: A list where each element is a tuple, the first item is the content of the news article, and the second item is the classification result of the news article.
    """
    data_path = "./bbc"
    categories = [('business', 0), ('entertainment', 1), ('politics', 2), ('sport', 3), ('tech', 4)]
    full_dataset = []
    for category in categories:
        category_path = os.path.join(data_path, category[0])
        for new in os.listdir(category_path):
            if new.endswith('.txt'):
                with open(os.path.join(category_path, new), encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    full_dataset.append((content, category[1]))
    return full_dataset

def get_word_list_in_sentence(string):
    """
    Function: Data preprocessing.
    :param string: A news article.
    :return: Individual word lists after word splitting, lowercase transfer, removing non-literal information, and removing deactivated words.
    """
    # Initialize lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()
    # Split text into sentences using NLTK
    sentence_split = nltk.tokenize.sent_tokenize(string)
    word_list = []
    for sentence in sentence_split:
        #  Using regular expression to choose word, and convert to lowercase
        pre_word_list = re.findall(r'\b\w+\b', sentence.lower())
        for word in pre_word_list:
            word_list.append(lemmatizer.lemmatize(word))
    return word_list

def create_frequency_dictionary(full_dataset, max_num_features):
    """
    Function: Creating a Global Dictionary.
    :param full_dataset: Global dataset.
    :param max_num_features: Maximum number of words in the frequency dictionary.
    :return: Frequency dictionary.
    """
    # Splitting words, eliminating deactivated words, counting frequencies
    word_count_list = {}
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords.update(['.', ',', "--", "``", ":", "\n", "'", '"', "''", "`", "´", "-", "—"])

    for new_and_category in full_dataset:
        word_list = get_word_list_in_sentence(new_and_category[0])
        for word in word_list:
            if word in stopwords:
                continue
            elif word in word_count_list:
                word_count_list[word] += 1
            else:
                word_count_list[word] = 1
    # Sorting, creating dictionaries
    sorted_list = sorted(word_count_list.items(), key=operator.itemgetter(1), reverse=True)[:max_num_features]
    dictionary = []
    for word, frequency in sorted_list:
        dictionary.append(word)
    return dictionary

def get_frequency_vector(dictionary, string):
    """
    Function: Processing inputs according to the frequency dictionary.
    :param dictionary: Frequency dictionary.
    :param string: The text string to be processed.
    :return: Processed text vector
    """
    word_vector = np.zeros(len(dictionary))
    word_list = get_word_list_in_sentence(string)
    for index, word in enumerate(dictionary):
        if word in word_list:
            word_vector[index] = word_list.count(word)  # Count the number of times the word appears in the sentence
    return word_vector

def word_embedding_training(dataset):
    """
    Function: Training the Word2Vec model.
    :param dataset: Dataset.
    :return: Trained model
    """
    sentences = []
    for new in dataset:
        word_list = get_word_list_in_sentence(new[0])
        sentences.append(word_list)
    word2Vec_model = Word2Vec(sentences, vector_size=100, window=5, seed=42)
    return word2Vec_model

def get_embedding_vector(word2Vec_model, string):
    """
    Compute the word embedding feature vector for the given text.
    :param string: String of text.
    :param word2Vec_model: Trained Word2Vec model.
    :return: Word embedding feature vector for a given text.
    """
    dictionary = word2Vec_model.wv
    word_list = get_word_list_in_sentence(string)
    embedding_vector = np.zeros(100)
    num = 0

    for word in word_list:
        if word in dictionary:
            embedding_vector += dictionary[word]
            num += 1
    if num:
        embedding_vector /= num
    return embedding_vector

def get_sentiment_score(string):
    """
    Calculating the sentiment score of input text by analyzing each sentence.
    :param string: input text.
    :return: Overall sentiment score of the input text.
    """
    sia = SentimentIntensityAnalyzer()
    sentences_list = nltk.tokenize.sent_tokenize(string)
    sum_score = 0
    sentiment_scores_original = []
    sentiment_scores_with_weight = 0
    # Calculate sentiment score with weights
    for sentence in sentences_list:
        score_original = sia.polarity_scores(sentence)
        sentiment_scores_original.append(score_original['compound'])
        sum_score += abs(score_original['compound'])
    if sum_score == 0:
        return 0
    for score in sentiment_scores_original:
        weight = abs(score) / sum_score
        sentiment_scores_with_weight += score * weight
    return sentiment_scores_with_weight

def get_combined_vector(frequency_dictionary, word2Vec_model, dataset):
    """
    Combine multiple features in a given dataset.
    :param frequency_dictionary: Frequency_dictionary.
    :param word2Vec_model: Trained Word2Vec model.
    :param dataset: The given dataset.
    :return:
    """
    X = []
    Y = []
    for index, new in enumerate(dataset):
        frequency_vector = get_frequency_vector(frequency_dictionary, new[0])
        embeddings_vector = get_embedding_vector(word2Vec_model, new[0])
        sentiment_score = np.array([get_sentiment_score(new[0])])
        combined_vector = np.concatenate((frequency_vector, embeddings_vector, sentiment_score))
        X.append(combined_vector)
        Y.append(new[1])
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y

def log_reg_muti_training(dataset, frequency_dictionary, word2Vec_model):
    """
    Function: Training logistic regression models
    :param dataset: Specified data set.
    :param frequency_dictionary: Frequency dictionary.
    :return: Well-trained logistic regression classifiers.
    """
    # Building the training set
    print("Grid search is starting...")
    X, Y = get_combined_vector(frequency_dictionary, word2Vec_model, dataset)
    # Machine learning pipeline
    pipeline = Pipeline([
        ("feature_selection", SelectKBest(score_func=mutual_info_classif)),
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=42)),
        ("logistic", LogisticRegression(multi_class="multinomial", solver='lbfgs', random_state=42))
    ])
    # Parameters space for grid search
    parameters_sapce = {
        "feature_selection__k": [50, 100, 150, 200],
        "pca__n_components": [0.8, 0.85, 0.9, 0.95],
        "logistic__C": [0.1, 0.4, 0.7, 1]
    }
    # Grid search with k-fold
    cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)
    grid_search = GridSearchCV(pipeline, parameters_sapce, cv=cv, scoring="f1_macro", n_jobs=-1)
    best_model = grid_search.fit(X, Y)
    print(f"The best parameter: {grid_search.best_params_}")
    print(f"In training set, the best macro averaged f1-score is: {round(grid_search.best_score_, 3)}")
    return best_model

def kfold_training(training_and_dev_set, k):
    """
    Function: k-fold cross-validation, and tuning parameter by grid search.
    :param training_and_dev_set: The training and development sets of data.
    :param k: The k value of k-fold.
    :return: The best classifier with the best parameters is selected after grid search, and its frequency dictionary.
    """
    best_model = None
    highest_f1 = 0
    best_frequency_dictionary = []
    word2Vec_model = word_embedding_training(training_and_dev_set)
    # Create a k-fold cross-validator and set k
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    # Directly generate a list of indexes for all k-fold cases of training and test set data.
    all_k_index_list_of_training_set_and_dev_set = kfold.split(training_and_dev_set)
    # Training and verifying by each k of iterations
    f1_all = 0
    accuracy_all = 0
    precision_all = 0
    recall_all = 0
    loop_count = 0
    print("K-Fold cross-validation in progress...")
    for this_k_training_set_index_list, this_k_dev_set_index_list in all_k_index_list_of_training_set_and_dev_set:
        loop_count += 1
        print(f"\nIn batch {loop_count}")
        training_set = []
        dev_set = []
        # (1) Create a list of training and test sets under this k
        for index, new in enumerate(training_and_dev_set):
            if index in this_k_training_set_index_list:
                training_set.append(new)
            else:
                dev_set.append(new)
        # (2) Training the model under this k
        # Create a global dictionary of reserved keywords under this k
        frequency_dictionary = create_frequency_dictionary(training_set, 1500)
        # Training the model under this k
        kfold_log_reg_muti_model = log_reg_muti_training(training_set, frequency_dictionary, word2Vec_model)

        # (3) Use f1_score to validate the performance of the model under this k
        X_dev, Y_dev_gold = get_combined_vector(frequency_dictionary, word2Vec_model, dev_set)
        Y_dev_predictions = kfold_log_reg_muti_model.predict(X_dev)
        # Computing four values
        f1 = f1_score(Y_dev_gold, Y_dev_predictions, average='macro')
        accuracy = accuracy_score(Y_dev_gold, Y_dev_predictions)
        macro_averaged_precision = precision_score(Y_dev_gold, Y_dev_predictions, average='macro')
        macro_averaged_recall = recall_score(Y_dev_gold, Y_dev_predictions, average='macro')
        # Select best f1, model and frequency dictionary
        if f1 > highest_f1:
            best_model = kfold_log_reg_muti_model
            highest_f1 = f1
            best_frequency_dictionary = frequency_dictionary
        print(f"In development set, Macro averaged f1-score is: {round(f1, 3)}")
        # Computing the sum values
        f1_all += f1
        accuracy_all += accuracy
        precision_all += macro_averaged_precision
        recall_all += macro_averaged_recall
    # Computing four average values to show the results
    f1_average = round(f1_all / k, 3)
    accuracy_average = round(accuracy_all/k, 3)
    precision_average = round(precision_all/k, 3)
    recall_average = round(recall_all/k, 3)
    print("\nThe results in cross-validation:")
    print(f"The macro averaged precision is: {precision_average}")
    print(f"The macro averaged recall is: {recall_average}")
    print(f"The macro averaged F1-Score is: {f1_average}")
    print(f"Accuracy: {accuracy_average}")
    return best_model, best_frequency_dictionary, word2Vec_model

# 2 running
full_dataset = load_data()
training_and_dev_set, test_set = train_test_split(full_dataset, test_size=0.2, random_state=42, shuffle=True)
clf_model, best_frequency_dictionary, word2Vec_model = kfold_training(training_and_dev_set, 5)

# 3 test
X_test, Y_test_gold = get_combined_vector(best_frequency_dictionary, word2Vec_model, test_set)
Y_test_predictions = clf_model.predict(X_test)
accuracy = accuracy_score(Y_test_gold, Y_test_predictions)
macro_averaged_precision = precision_score(Y_test_gold, Y_test_predictions, average='macro')
macro_averaged_recall = recall_score(Y_test_gold, Y_test_predictions, average='macro')
macro_averaged_f1 = f1_score(Y_test_gold, Y_test_predictions, average='macro')

print("\n===================================================================")
print("The results in the test set:")
print(f"Macro averaged precision: {round(macro_averaged_precision, 3)}")
print(f"Macro averaged recall: {round(macro_averaged_recall, 3)}")
print(f"Macro averaged F1-Score: {round(macro_averaged_f1, 3)}")
print(f"Accuracy: {round(accuracy, 3)}")
print("Classification Report:")
print(classification_report(Y_test_gold, Y_test_predictions))
