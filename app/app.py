import os
from flask import Flask, request, render_template, jsonify
import joblib
import scipy.sparse as sp

from preprocessing import preprocessing

app = Flask(__name__)

model_path = os.path.join('models', 'svm_rbf_model.pkl')
model_pos_path = os.path.join('models', 'svm_rbf_model_pos.pkl')
model_neg_path = os.path.join('models', 'svm_rbf_model_neg.pkl')
vectorizer_path = os.path.join('vectorizers', 'tfidf_vectorizer.joblib')
vectorizer_pos_path = os.path.join('vectorizers', 'tfidf_vectorizer_pos.joblib')
vectorizer_neg_path = os.path.join('vectorizers', 'tfidf_vectorizer_neg.joblib')

model = joblib.load(model_path)
model_pos = joblib.load(model_pos_path)
model_neg = joblib.load(model_neg_path)
vectorizer = joblib.load(vectorizer_path)
vectorizer_pos = joblib.load(vectorizer_pos_path)
vectorizer_neg = joblib.load(vectorizer_neg_path)


@app.route('/')
def general():
    return render_template('general.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text = preprocessing(text)
    raw_text = [" ".join(text)]
    vectors = vectorizer.transform(raw_text)
    vectors = sp.csr_matrix(vectors)
    predict = model.predict(vectors)
    if predict == [1]:
        vectors = vectorizer_pos.transform(raw_text)
        vectors = sp.csr_matrix(vectors)
        predict = model_pos.predict(vectors)
        result = f"Положительный: {predict[0]}/10"
    else:
        vectors = vectorizer_neg.transform(raw_text)
        vectors = sp.csr_matrix(vectors)
        predict = model_neg.predict(vectors)
        result = f"Отрицательный: {predict[0]}/10"
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=False)
