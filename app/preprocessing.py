import re
from nltk.corpus import stopwords
import spacy


def preprocessing(comment: str) -> list[str] | None:
    text = remove_url(comment)
    text = remove_tags(text)
    text = remove_username(text)
    text = remove_id(text)
    text = remove_brackets(text)
    text = remove_punct_numbers(text)
    text = lowercase_text(text)
    tokens = tokens_lemmatize(text)
    tokens = remove_stopwords(tokens)
    tokens = removing_short_tokens(tokens)
    if tokens != []:
        preprocessed_comment_text = tokens
        return preprocessed_comment_text


def remove_url(text: str) -> str:
    text = re.sub(r'https?://\S+', '', text)
    return text


def remove_tags(text: str) -> str:
    text = re.sub(r'#\S+', '', text)
    return text


def remove_username(text: str) -> str:
    text = re.sub(r'@\S+', '', text)
    return text


def remove_brackets(text: str) -> str:
    text = re.sub(r'\<[a-zA-Z]+\>', ' ', text)
    return text


def remove_id(text: str) -> str:
    text = re.sub(r'id', '', text)
    return text


def remove_punct_numbers(text: str) -> str:
    text = re.sub(r'[^\w\s]|[\d]', '', text)
    return text


def lowercase_text(text: str) -> str:
    text = text.lower()
    return text


def remove_stopwords(tokens: list[str]) -> list[str]:
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


def tokens_lemmatize(text: str) -> list[str]:
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return lemmas


def removing_short_tokens(tokens: list[str]) -> list[str]:
    tokens = [token for token in tokens if len(token) > 2]
    return tokens
