import pickle
import pandas as pd


class Utils:
    @staticmethod
    def write_object_to_file(obj, file_name):
        with open(file_name, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def read_object_from_file(file_name):
        with open(file_name, 'rb') as input:
            return pickle.load(input)

    @staticmethod
    def generate_submission_csv(test_labels):
        """
        This method genertes a CSV file from a
        :param test_labels:
        :return:
        """
        df_test = pd.DataFrame(test_labels)
        df_test.columns = ["Category"]
        df_test.to_csv("submission.csv")

    @staticmethod
    def split(n):
        """
        This method splits the train data into two sets: train, validation
        :param n:
        :return:
        """
        df = pd.read_pickle("data_train.pkl")
        data = {'post': list(df)[0], 'class': list(df)[1]}
        df_train = pd.DataFrame(data)
        df_train = df_train.sample(frac=1).reset_index(drop=True)  # shuffle the dataset
        length = df_train.__len__()
        validation_size = int(length * n / 100)
        df_train = df_train.head(length - validation_size)
        df_validation = df_train.head(validation_size)
        return df_train, df_validation

    @staticmethod
    def score(predictions, result):
        """
        This method calculates the score of the model providing the prediction and actual results
        :param predictions:
        :param result:
        :return: prediction
        """
        count = 0
        for i in range(len(predictions)):
            if (predictions[i] == result[i]):
                count += 1
        return count / len(predictions)

    @staticmethod
    def get_train_data():
        """
        This method extracts the training data from a pkl file as a dataframe
        :return:
        """
        df = pd.read_pickle("data_train.pkl")
        data = {'post': list(df)[0], 'class': list(df)[1]}
        return pd.DataFrame(data)

    @staticmethod
    def get_test_data():
        """
        This method extracts the test data from a pkl file as a dataframe
        :return:
        """
        df_test = pd.read_pickle("data_test.pkl")
        data_test = {'post': list(df_test)}
        return pd.DataFrame(data_test)
