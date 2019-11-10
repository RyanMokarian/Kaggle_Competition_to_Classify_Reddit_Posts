import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from src.utils import Utils

from nltk.corpus import stopwords

nltk.download('stopwords')
english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])


class PreBuiltClassifier:
    classes = {'hockey': 0, 'nba': 1, 'leagueoflegends': 2, 'soccer': 3, 'funny': 4, 'movies': 5, 'anime': 6,
               'Overwatch': 7, 'trees': 8, 'GlobalOffensive': 9,
               'nfl': 10, 'AskReddit': 11, 'gameofthrones': 12,
               'conspiracy': 13, 'worldnews': 14, 'wow': 15, 'europe': 17, 'canada': 18, 'Music': 19, 'baseball': 20}
    utils = Utils()
    stopwords = stopwords.words('english')

    def naive_bayes_sickit(self, df_train):
        """
        alpha=0.005 -> 0.53
        alpha=0.00005 -> 0.51
        alpha= 0.5 -> 0.547
        0.5434857142857142
        tf_idf:
        alpha =0.5 -> 0.56
        alpha= 0.005 -> 0.7620571428571429
        :param train:
        :param test
        :return:
        """

        from sklearn.naive_bayes import MultinomialNB
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(df_train['post'], df_train['class'], random_state=53)

        nb = Pipeline([('vect', StemmedCountVectorizer(
            stop_words='english',  # works
        )),
                       ('tfidf', TfidfTransformer(sublinear_tf=True, smooth_idf=False)),

                       ('clf', MultinomialNB(alpha=0.25)),
                       ])

        nb.fit(X_train, y_train)

        scores = cross_val_score(nb, X_train, y_train, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        preds = nb.predict(X_test)
        UTILS.generate_submission_csv(preds)
        print("Accuracy against test data:" + str(accuracy_score(y_test, preds)))

    def linear_svc(self, df):
        # 0.4758008658008658
        # preprocessed : 0.47935064935064936

        from sklearn.feature_extraction.text import CountVectorizer
        model = LinearSVC()

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(df['post'], df['class'],
                                                                                         df.index,
                                                                                         test_size=0.33, random_state=0)
        vect = CountVectorizer().fit(X_train)
        # X_train_vectorized = vect.transform(X_train)
        # model.fit(X_train_vectorized, y_train)
        nb = Pipeline([('vect', StemmedCountVectorizer(stop_words='english')),
                       ('tfidf', TfidfTransformer(sublinear_tf=True, smooth_idf=False)),
                       ('clf', LinearSVC()),
                       ])
        nb.fit(X_train, y_train)
        y_pred = nb.predict(vect.transform(X_test))
        # UTILS.generate_submission_csv(y_pred)
        print(accuracy_score(y_test, y_pred))

    def logistic_regression(self, df):
        """
        penalty "l2" -> 0.5415584415584416 cross validation  (CV=5) Accuracy: 0.54 (+/- 0.01)
        penalty "l1" -> :0.5235930735930736 cross validation  (CV=5) Accuracy: 0.52 (+/- 0.01)
        :param df:
        :return:
        """

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(df['post'], df['class'],
                                                                                         df.index,
                                                                                         test_size=0.33, random_state=0)
        nb = Pipeline([('vect', StemmedCountVectorizer(stop_words='english')),
                       ('tfidf', TfidfTransformer(sublinear_tf=True, smooth_idf=False)),
                       ('logistic-regression', LogisticRegression(penalty="l2")),
                       ])
        nb.fit(X_train, y_train)
        scores = cross_val_score(nb, X_train, y_train, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        preds = nb.predict(X_test)
        # UTILS.generate_submission_csv(preds)
        print("Accuracy against test data:" + str(accuracy_score(y_test, preds)))

    def mlp_predict_test_to_file(self, df_train, df_test):
        nb = Pipeline([('vect', CountVectorizer(stop_words=self.stopwords)),
                       ('tfidf', TfidfTransformer(sublinear_tf=True, smooth_idf=False)),
                       ('clf', MLPClassifier(verbose=True, early_stopping=True)),
                       ])
        nb.fit(df_train['post'], df_train['class'])

        y_pred = nb.predict(df_test['post'])
        UTILS.generate_submission_csv(y_pred)


if __name__ == "__main__":
    UTILS = Utils()
    PRE_BUILD_CLASSIFIER = PreBuiltClassifier()

    df_train = UTILS.get_train_data()
    df_test = UTILS.get_test_data()

    PRE_BUILD_CLASSIFIER.mlp_predict_test_to_file(df_train, df_test)
