"""
alpha=0.05 (30-70 data divide) accuracy -> 0.7982857142857143
alpha=0.005 (30-70 data divide) accuracy ->0.8183333333333334
alpha=0.0005 (30-70 data divide) accuracy ->0.8237142857142857
"""
import string

import operator
import time
import nltk
from nltk.corpus import stopwords
import numpy as np
from nltk.stem import WordNetLemmatizer

from src.utils import Utils

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class NaiveBayes:
    """
    This Class contains the modules for Naive Bayes Classifier
    """

    def __init__(self, alpha=0.5):
        self.stopwords = stopwords.words('english') + list(string.punctuation)
        self.classes = []
        self.total_words_in_each_class = {}
        self.alpha = alpha
        self.count_of_all_words_in_corpus = 0
        self.priors = {}

    def predict(self, df_test):
        """
        This method contains the processes that needs to be done to predict the classes for the test data
        It loads the predicted data (p_t_give_c.pkl) and applies Bye theorem to predict the classes
        and write them into a CSV file
        :return:
        """
        probability_of_t_given_c_all_classes = Utils.read_object_from_file("p_t_give_c.pkl")
        classes_test = []
        x = 0
        for index, row in df_test.iterrows():
            x += 1
            tokenized = nltk.word_tokenize(row["post"])
            tokens = self.apply_pre_processing(tokenized)
            classes_test.append(
                self.compute_max_likelihood_for_all_classes(probability_of_t_given_c_all_classes,
                                                            tokens))
            if x % 50 == 0:
                print(str(x) + " items predicted")
        return classes_test

    def compute_max_likelihood_for_all_classes(self, probability_of_t_given_c_all_classes,
                                               tokens):
        """
        This method calculates the maximum likelihood per class
        :param probability_of_t_given_c_all_classes:
        :param tokens:
        :return: class having the maximum probability
        """
        probability_of_terms_belonging_to_classes = {}
        for a_class in self.classes:
            probability_of_terms_belonging_to_classes[a_class] = self.compute_likelihood_for_this_class(
                probability_of_t_given_c_all_classes[a_class],
                self.priors[a_class], tokens, a_class)

        return max(probability_of_terms_belonging_to_classes.items(), key=operator.itemgetter(1))[0]

    def compute_likelihood_for_this_class(self, p_of_term_belonging_to_class,
                                          prior,
                                          tokens,
                                          a_class):
        """
        log p(c|d) =  log(p(x1,x2,...,xn|c) + log p(c)   (note the denominator of p(d) is disregarded)
        WHERE log p(x1,x2,...,xn|c) = log p(x1|c) + log p(x2|c)+ ... + log p(xn|c)
        prior: p(c)
        if there are unseen tokens, it gives it a probability equal to :
        p= alpha /(#words_in_corpus  * alpha) + # words_in_this_class
        :return:
        """
        p_xi_given_c = []
        for token in tokens:
            if token in p_of_term_belonging_to_class:
                p_xi_given_c.append(p_of_term_belonging_to_class[token])
            else:
                denominator = np.add(np.multiply(float(self.count_of_all_words_in_corpus), self.alpha),
                                     float(self.total_words_in_each_class[a_class]))
                probability = np.divide(self.alpha, denominator)
                p_xi_given_c.append(probability)
        sum_p_X_given_c = 0.0
        for p in p_xi_given_c:
            sum_p_X_given_c += np.log(p)
        log_p_X_given_c_plus_log_prior = sum_p_X_given_c + np.log(prior)

        return log_p_X_given_c_plus_log_prior

    def fit(self, df_train):
        """
        This function computes probability of terms for all classes and store them into a file
        :return:
        """
        self.priors = df_train["class"].value_counts() / len(df_train)
        self.classes = df_train["class"].unique()
        term_frequency_per_class = self.calculate_tf_and_sum_of_all_docs(df_train)
        self.count_of_all_words_in_corpus = len(term_frequency_per_class)

        probability_of_t_given_c_all_classes = self.compute_probability_of_terms_for_all_classes(
            term_frequency_per_class)
        Utils.write_object_to_file(probability_of_t_given_c_all_classes, "p_t_give_c.pkl")

    @staticmethod
    def get_total_words_in_class(term_frequency_per_class, a_class):
        """
        This method calculates the total number of words in a class
        :param term_frequency_per_class:
        :param a_class:
        :return:
        """
        sum = 0
        for term in term_frequency_per_class:
            sum += term_frequency_per_class[term][a_class]
        return sum

    def compute_probability_of_terms_for_all_classes(self, term_frequency_per_class):
        """
        This function computes probability of terms for all classes
        p(t|c) = (# of frequency of the term in class c) / (total number of words in this class)
        {"hockey":  {"yoga":"0.009", "football":"0.0082" ....},
        "nba": .....
        }
        """

        probability_of_t_given_c_all_classes = {}
        for a_class in self.classes:
            total_number_of_words_in_c = self.get_total_words_in_class(term_frequency_per_class, a_class)
            self.total_words_in_each_class[a_class] = total_number_of_words_in_c
            probability_of_t_given_c = {}
            for term in term_frequency_per_class:
                term_frequency_in_documents = term_frequency_per_class[term][a_class]
                nominator = np.add(float(term_frequency_in_documents), self.alpha)
                denominator = np.add(np.multiply(float(self.count_of_all_words_in_corpus), self.alpha),
                                     float(total_number_of_words_in_c))
                probability = np.divide(nominator, denominator)
                probability_of_t_given_c[term] = probability

            probability_of_t_given_c_all_classes[a_class] = probability_of_t_given_c
            print("priors for class:" + str(a_class) + " is computed")
        return probability_of_t_given_c_all_classes

    def calculate_tf_and_sum_of_all_docs(self, df_train):
        """
        Calculating all the rows in the corups (after pre_processing )
        We keep duplicates intentionally because in compute_probability_of_term_occurrence_in_this_class
        we also have duplicates and iterating over a list of words in that class
        :param df_train:
        :return:
        """
        term_frequency_per_class = {}
        for index, row in df_train.iterrows():
            current_class = row["class"]
            tokenized = nltk.word_tokenize(row["post"])
            pre_processed_words = self.apply_pre_processing(tokenized)
            for word in pre_processed_words:
                if word not in term_frequency_per_class:
                    term_frequency_per_class[word] = {}
                for a_class in self.classes:
                    if a_class != current_class:
                        if a_class not in term_frequency_per_class[word]:
                            term_frequency_per_class[word][a_class] = 0
                    else:
                        if a_class not in term_frequency_per_class[word]:
                            term_frequency_per_class[word][a_class] = 1
                        else:
                            term_frequency_per_class[word][a_class] += 1

            if index % 500 == 0:
                print(str(index) + " records in training data processed")

        return term_frequency_per_class

    def apply_pre_processing(self, a_list):
        """
        This method applies processing to a list of words
        Applying:
        casefolding, removing numbers, removing stop words and applying lemmatization
        :param a_list:
        :return: a list of processed words
        """
        pre_processed = list()
        lemmatizer = WordNetLemmatizer()
        for word in a_list:
            word = word.lower()
            if not word.isalpha():
                continue
            if word in self.stopwords:
                continue
            word = lemmatizer.lemmatize(word)
            pre_processed.append(word)

        return pre_processed


def verify_with_train_data():
    """
    This method split the training data into train, validation to verify the model performance locally
    :return:
    """
    NAIVE_BAYES = NaiveBayes(1)
    start_time = time.time()
    df_train, df_validation = Utils.split(30)
    NAIVE_BAYES.fit(df_train)

    classes_test = NAIVE_BAYES.predict(df_validation)
    print(classes_test)
    score = Utils.score(np.array(classes_test), df_validation["class"].to_numpy())
    print("Score is equal to: " + str(score))
    print("--- %s seconds ---" % (time.time() - start_time))


def run_classifier_and_generate_csv():
    """
    This method runs the model with training and test data and generates a CSV file with predicted classes
    :return:
    """

    start_time = time.time()
    UTILS = Utils()
    NAIVE_BAYES = NaiveBayes(1)
    df_train = UTILS.get_train_data()
    NAIVE_BAYES.fit(df_train)

    TEST_CLASSES = NAIVE_BAYES.predict(UTILS.get_test_data())
    UTILS.generate_submission_csv(TEST_CLASSES)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    run_classifier_and_generate_csv()
