"""
alpha =1, (50-50 data divide) accuracy ->   0.84%
alpha =0.95, (50-50 data divide) accuracy ->   0.84%
alpha = 0.85, (50-50 data divide) accuracy ->   0.839%
alpha =0.75, (50-50 data divide) accuracy ->   0.83%
alpha =0.5, (50-50 data divide) accuracy ->   0.83%
alpha =0.05, (50-50 data divide) accuracy ->   0.82%
"""
import string

import operator
import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np
from nltk.stem import WordNetLemmatizer

from src.utils import write_object_to_file, read_object_from_file

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stopwords = stopwords.words('english') + list(string.punctuation)

classes = ['hockey', 'nba', 'leagueoflegends', 'soccer', 'funny', 'movies', 'anime',
           'Overwatch', 'trees', 'GlobalOffensive',
           'nfl', 'AskReddit', 'gameofthrones',
           'conspiracy', 'worldnews', 'wow', 'europe', 'canada', 'Music', 'baseball']


def test(df_test, priors):
    """
    This method contains the processes that needs to be done to predict the classes for the test data
    It loads the predicted data (p_t_give_c.pkl) and applies Bye theorem to predict the classes
    and write them into a CSV file
    :return:
    """
    probability_of_t_given_c_all_classes = read_object_from_file("p_t_give_c.pkl")
    classes_test = []
    x = 0
    for index, row in df_test.iterrows():
        x += 1
        tokenized = nltk.word_tokenize(row["post"])
        tokens = apply_pre_processing(tokenized)
        classes_test.append(
            compute_max_likelihood_for_all_classes(probability_of_t_given_c_all_classes,
                                                   priors,
                                                   tokens))
        if x % 50 == 0:
            print(str(x) + " items processed")
    return classes_test


def compute_max_likelihood_for_all_classes(probability_of_t_given_c_all_classes,
                                           priors,
                                           tokens):
    probability_of_terms_belonging_to_classes = {}
    for a_class in classes:
        probability_of_terms_belonging_to_classes[a_class] = compute_likelihood_for_this_class(
            probability_of_t_given_c_all_classes[a_class],
            priors[a_class], tokens)

    return max(probability_of_terms_belonging_to_classes.items(), key=operator.itemgetter(1))[0]


def compute_likelihood_for_this_class(p_of_term_belonging_to_class, prior, tokens):
    # S
    # Here log formula can be used
    # E
    """
    log p(c|d) =  log(p(x1,x2,...,xn|c) + log p(c)   (note the denominator of p(d) is disregarded)
    WHERE log p(x1,x2,...,xn|c) = log p(x1|c) + log p(x2|c)+ ... + log p(xn|c)
    prior: p(c)
    :return:
    """
    p_xi_given_c = []
    for token in tokens:
        if token in p_of_term_belonging_to_class:
            p_xi_given_c.append(p_of_term_belonging_to_class[token])
        else:
            p_xi_given_c.append(
                0)  # never seen feature and class in training data https://en.wikipedia.org/wiki/Naive_Bayes_classifier
    # p_X_given_c = np.prod(p_xi_given_c)
    # p_X_given_c_mult_prior = np.multiply(p_X_given_c, prior)
    sum_p_X_given_c = 0.0
    for p in p_xi_given_c:
        sum_p_X_given_c += np.log(p)
    log_p_X_given_c_plus_log_prior = sum_p_X_given_c + np.log(prior)

    return log_p_X_given_c_plus_log_prior


def train(df_train):
    """
    This function computes probability of terms for all classes and store them into a file
    :return:
    """
    count_of_all_words_in_corpus, term_frequency_per_class = calculate_tf_and_sum_of_all_docs(df_train)
    # count_of_all_words_in_corpus = 33847
    print("All teh words in the corpus is " + str(count_of_all_words_in_corpus))
    probability_of_t_given_c_all_classes = compute_probability_of_terms_for_all_classes(term_frequency_per_class,
                                                                                        count_of_all_words_in_corpus)
    write_object_to_file(probability_of_t_given_c_all_classes, "p_t_give_c.pkl")


def compute_probability_of_terms_for_all_classes(term_frequency_per_class, count_of_all_words_in_corpus):
    """
    This function computes probability of terms for all classes
#S
    p(t|c) = (# of frequency of the term in class c) / (total number of words in this class)
#E
    {"hockey":  {"yoga":"0.009", "football":"0.0082" ....},
    "nba": .....
    }
    """

    probability_of_t_given_c_all_classes = {}
    for a_class in classes:
        total_number_of_words_in_c = sum(term_frequency_per_class[a_class].values())
        probability_of_t_given_c = {}
        for term in term_frequency_per_class[a_class]:
            term_frequency_in_documents = term_frequency_per_class[a_class][term]
            alpha = 1.0
            nominator = np.add(float(term_frequency_in_documents), alpha)
            denominator = np.add(np.multiply(float(total_number_of_words_in_c), alpha),
                                 float(count_of_all_words_in_corpus))
            probability = np.divide(nominator, denominator)
            probability_of_t_given_c[term] = probability

        probability_of_t_given_c_all_classes[a_class] = probability_of_t_given_c
        print("class:" + str(a_class) + " is Done!")
    return probability_of_t_given_c_all_classes


def calculate_tf_and_sum_of_all_docs(df_train):
    """
    Calculating all the rows in the corups (after pre_processing )
    We keep duplicates intentionally because in compute_probability_of_term_occurrence_in_this_class
    we also have duplicates and iterating over a list of words in that class
    :param df_train:
    :return:
    """
    set_of_all_words_in_corpus = set()
    term_frequency_per_class = {}
    for index, row in df_train.iterrows():
        current_class = row["class"]
        tokenized = nltk.word_tokenize(row["post"])
        pre_processed_words = apply_pre_processing(tokenized)
        set_of_all_words_in_corpus.update(pre_processed_words)
        for word in pre_processed_words:
            if current_class not in term_frequency_per_class:
                term_frequency_per_class[current_class] = {}
            elif word not in term_frequency_per_class[current_class]:
                term_frequency_per_class[current_class][word] = 1
            else:
                term_frequency_per_class[current_class][word] += 1

        if index % 500 == 0:
            print("Index #" + str(index) + " is processed to the global terms to calculate V")

    return len(set_of_all_words_in_corpus), term_frequency_per_class


def apply_pre_processing(a_list):
    pre_processed = list()
    # porter = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    for word in a_list:
        word = word.lower()
        if not word.isalpha():
            continue
        if word in stopwords:
            continue
        # word = porter.stem(word)
        word = lemmatizer.lemmatize(word)
        pre_processed.append(word)

    return pre_processed


def compute_prior_for_each_class(df_train):
    return df_train["class"].value_counts() / len(df_train)


def split(n):
    df = pd.read_pickle("data_train.pkl")
    data = {'post': list(df)[0], 'class': list(df)[1]}
    df_train = pd.DataFrame(data)
    df_train = df_train.sample(frac=1).reset_index(drop=True)  # shuffle the dataset
    length = df_train.__len__()
    validation_size = int(length * n / 100)
    df_train = df_train.head(length - validation_size)
    df_validation = df_train.head(validation_size)
    return df_train, df_validation


def score(predictions, result):
    count = 0
    for i in range(len(predictions)):
        if (predictions[i] == result[i]):
            count += 1
    return count / len(predictions)


def get_train_dataframe():
    df = pd.read_pickle("data_train.pkl")
    data = {'post': list(df)[0], 'class': list(df)[1]}
    return pd.DataFrame(data)


def generate_submission_csv(test_labels):
    df_test = pd.DataFrame(test_labels)
    df_test.to_csv("submission.csv")


def get_test_data():
    df_test = pd.read_pickle("data_test.pkl")
    data_test = {'post': list(df_test)}
    return pd.DataFrame(data_test)


if __name__ == "__main__":
    """
    First run train() method
    Then Run test() method
    """
    # the following is just for our test to save our submissions

    import time

    start_time = time.time()
    df_train, df_validation = split(25)
    train(df_train)

    priors = compute_prior_for_each_class(df_train)
    classes_test = test(df_validation, priors)
    print(classes_test)
    score = score(np.array(classes_test), df_validation["class"].to_numpy())
    print("Score is equal to: " + str(score))
    print("--- %s seconds ---" % (time.time() - start_time))

    # uncomment below lines when you want to run a submission in Kaggle and comment out the lines above
    # start_time = time.time()
    # df_train = get_train_dataframe()
    # train(df_train)
    # df_test = get_test_data()
    # priors = compute_prior_for_each_class(df_train)
    # classes_test = test(df_test, priors)
    # generate_submission_csv(classes_test)
    # print("--- %s seconds ---" % (time.time() - start_time))
