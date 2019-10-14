import random

import pandas as pd

classes = ['hockey', 'nba', 'leagueoflegends', 'soccer', 'funny', 'movies', 'anime',
           'Overwatch', 'trees', 'GlobalOffensive',
           'nfl', 'AskReddit', 'gameofthrones',
           'conspiracy', 'worldnews', 'wow', 'europe', 'canada', 'Music', 'baseball']


def load_data():
    # df = pd.read_pickle("data_train.pkl")
    # data = {'post': list(df)[0], 'class': list(df)[1]}
    # df_train = pd.DataFrame(data)

    df_test = pd.read_pickle("data_test.pkl")
    data_test = {'category': list(df_test)}
    df_test = pd.DataFrame(data_test)
    generate_random_guess_csv()


def generate_random_guess_csv():
    lables = random.choices(classes, k=30000)
    df_test = pd.DataFrame(lables)
    df_test.to_csv("submission.csv")


    random.choices(classes)
    df_test = pd.DataFrame(lables)
    df_test.to_csv("submission.csv")


if __name__ == "__main__":
    generate_random_guess_csv()
