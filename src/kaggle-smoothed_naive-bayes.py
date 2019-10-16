import string

import operator
import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np
from nltk.stem.porter import PorterStemmer

from src.utils import write_object_to_file, read_object_from_file

nltk.download('punkt')
nltk.download('stopwords')

stopwords = stopwords.words('english') + list(string.punctuation)

classes = ['hockey', 'nba', 'leagueoflegends', 'soccer', 'funny', 'movies', 'anime',
           'Overwatch', 'trees', 'GlobalOffensive',
           'nfl', 'AskReddit', 'gameofthrones',
           'conspiracy', 'worldnews', 'wow', 'europe', 'canada', 'Music', 'baseball']
set_of_all_terms_in_all_classes = set()


def test():
    """
    This method contains the processes that needs to be done to predict the classes for the test data
    It loads the predicted data (p_t_give_c.pkl) and applies Bye theorem to predict the classes
    and write them into a CSV file
    :return:
    """
    priors = compute_priors()
    df_test = pd.read_pickle("data_test.pkl")
    data_test = {'category': list(df_test)}
    df_test = pd.DataFrame(data_test)
    probability_of_t_given_c_all_classes = read_object_from_file("p_t_give_c.pkl")
    classes_test = []
    x = 0
    for index, row in df_test.iterrows():
        x += 1
        tokenized = nltk.word_tokenize(row["category"])
        tokens = apply_preprocessing_on_list_of_text(tokenized)
        classes_test.append(
            compute_max_likelihood_for_all_classes(probability_of_t_given_c_all_classes,
                                                   priors,
                                                   tokens))
        if x % 50 == 0:
            print(str(x) + " items processed")
    df_test = pd.DataFrame(classes_test)
    df_test.to_csv("submission.csv")


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
    sum_p_X_given_c = 0
    for p in p_xi_given_c:
        sum_p_X_given_c+= np.log(p)
    log_p_X_given_c_plus_log_prior = sum_p_X_given_c + np.log(prior)

    return log_p_X_given_c_plus_log_prior


def compute_priors():
    """
    This document computes priors and total documents in each class
    :return:
    priors : # of documents belong to one class / total number of documents
    compute_total_documents_for_each_class : total number of documents in this class
    """
    df = pd.read_pickle("data_train.pkl")
    data = {'post': list(df)[0], 'class': list(df)[1]}
    df_train = pd.DataFrame(data)
    priors = compute_prior_for_each_class(df_train)
    return priors


def train():
    """
    This function computes probability of terms for all classes and store them into a file
    :return:
    """
    df = pd.read_pickle("data_train.pkl")
    data = {'post': list(df)[0], 'class': list(df)[1]}
    df_train = pd.DataFrame(data)
    probability_of_t_given_c_all_classes = compute_probability_of_terms_for_all_classes(df_train)
    write_object_to_file(probability_of_t_given_c_all_classes, "p_t_give_c.pkl")


def compute_probability_of_terms_for_all_classes(df_train):
    """
    This function computes probability of terms for all classes
#S
    p(t|c) = (# of frequency of the term in class c) / (total number of words in this class)
#E
    {"hockey":  {"yoga":"0.009", "football":"0.0082" ....},
    "nba": .....
    }
    """

# below only a set of words for all classes is computed.										   
    for a_class in classes:
        list_of_all_words_in_this_class, set_of_words_for_a_class = get_bag_of_words_for_a_class(a_class, df_train)
        set_of_all_terms_in_all_classes.update(set(set_of_words_for_a_class))
        print("class:" + str(a_class) + " is Done for getting global terms!")

    probability_of_t_given_c_all_classes = {}											 
    for a_class in classes:
# below, list_of_words added
        list_of_words, set_of_words, = get_bag_of_words_for_a_class(a_class, df_train)
# below set_of_words changed to list_of_words
        probability_of_t_given_c = compute_probability_of_term_occurrence_in_this_class(list_of_words, list_of_words)
        probability_of_t_given_c_all_classes[a_class] = probability_of_t_given_c
        print("class:" + str(a_class) + " is Done!")
    return probability_of_t_given_c_all_classes


def compute_probability_of_term_occurrence_in_this_class(list_of_words_in_a_class, set_of_words_in_a_class):
    """
    This function computes the p(t|c) for that specif class
#S
    p(t|c) = (# of frequency of the term in class c) / (total number of words in this class)
#E
    :param list_of_docs:
    :param set_of_words:
    :return: return type is similar to
    {"yoga":"0.009", "football":"0.0082" ....}
    """
    probability_of_t_given_c = {}
    total_number_of_words_in_c = len(list_of_words_in_a_class)
    for term in set_of_words_in_a_class:
        term_frequency_in_documents = 0
        for doc in list_of_words_in_a_class:
            if term in doc:
                term_frequency_in_documents += 1
        # probability_of_t_given_c[term] = term_frequency_in_documents / total_number_of_docs_in_c
        probability_of_t_given_c[term] = (term_frequency_in_documents + 1)/ (total_number_of_words_in_c + len(set_of_all_terms_in_all_classes))
    return probability_of_t_given_c


def get_bag_of_words_for_a_class(a_class, df_train):

    list_of_tokenized_documents_in_this_class = list()
    set_of_all_terms_in_this_class = set()
# below list added
    list_of_all_terms_in_this_class = list()
    for index, row in df_train.iterrows():
        if row["class"] == a_class:
            tokenized = nltk.word_tokenize(row["post"])
            pre_processed = apply_preprocessing_on_list_of_text(tokenized)
            set_of_all_terms_in_this_class.update(set(pre_processed))
# below line added
            list_of_all_terms_in_this_class.extend(pre_processed)

    return list_of_all_terms_in_this_class, set_of_all_terms_in_this_class


def apply_preprocessing_on_list_of_text(a_list):
    # case folding
    lower_case_list = [x.lower() for x in a_list]
    # keep alphabetic characters
    words = [word for word in lower_case_list if word.isalpha()]
    # remove punctuations and stopwords
    words = [w for w in words if not w in stopwords]
    # apply stemming
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    return stemmed


def compute_prior_for_each_class(df_train):
    return df_train["class"].value_counts() / len(df_train)


def compute_total_documents_for_each_class(df_train):
    return df_train["class"].value_counts()


if __name__ == "__main__":
    """
    First run train() method
    Then Run test() method
    """
    train()
    # test()
