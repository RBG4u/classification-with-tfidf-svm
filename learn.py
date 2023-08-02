import joblib
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
import scipy.sparse as sp


def make_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv('train_prep.csv', delimiter=',')
    test_df = pd.read_csv('test_prep.csv', delimiter=',')
    return train_df, test_df


def make_corpus(train: pd.DataFrame, test: pd.DataFrame
                ) -> tuple[pd.Series, pd.Series]:
    train['Text'] = train['Text'].apply(make_dict)
    test['Text'] = test['Text'].apply(make_dict)
    train['Text'] = train['Text'].apply(connect_to_str)
    test['Text'] = test['Text'].apply(connect_to_str)
    corpus_train = train['Text'].tolist()
    corpus_test = test['Text'].tolist()
    return corpus_train, corpus_test


def make_dict(str: str) -> list[str]:
    dictionary = ast.literal_eval(str)
    return dictionary


def prepr_score(train: pd.DataFrame, test: pd.DataFrame
                ) -> tuple[pd.Series, pd.Series]:
    score_test = test['Score'].apply(make_binary)
    score_train = train['Score'].apply(make_binary)
    return score_train, score_test


def connect_to_str(my_list: list[str]) -> str:
    result_string = ' '.join(my_list)
    return result_string


def make_vectors(corpus_train: pd.Series, corpus_test:  pd.Series
                 ) -> tuple[sp.csc_matrix, sp.csr_matrix]:
    vectorizer = TfidfVectorizer()

    vec_train = vectorizer.fit_transform(corpus_train)
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    vec_test = vectorizer.transform(corpus_test)
    vec_train = sp.csr_matrix(vec_train)
    vec_test = sp.csr_matrix(vec_test)
    return vec_train, vec_test


def make_binary(value: int) -> int:
    if value > 5:
        return 1
    else:
        return -1


def make_pos_data(train_df: pd.DataFrame, test_df: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexes = train_df[train_df['Score'] < 5].index
    indexes_test = test_df[test_df['Score'] < 5].index

    train_df.drop(indexes, inplace=True)
    test_df.drop(indexes_test, inplace=True)
    return train_df, test_df


def make_neg_data(train_df: pd.DataFrame, test_df: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexes = train_df[train_df['Score'] > 5].index
    indexes_test = test_df[test_df['Score'] > 5].index

    train_df.drop(indexes, inplace=True)
    test_df.drop(indexes_test, inplace=True)
    return train_df, test_df


def learn_model() -> None:
    train_df, test_df = make_data()
    corpus_train, corpus_test = make_corpus(train_df, test_df)
    score_train, score_test = prepr_score(train_df, test_df)
    vec_train, vec_test = make_vectors(corpus_train, corpus_test)

    model = svm.SVC(kernel='rbf', C=15, gamma='scale')
    model.fit(vec_train, score_train)
    pred = model.predict(vec_test)
    accuracy = accuracy_score(score_test, pred)
    print("Accuracy:", accuracy)
    joblib.dump(model, 'svm_rbf_model.pkl')


if __name__ == "__main__":
    learn_model()
