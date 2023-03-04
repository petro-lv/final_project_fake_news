from nltk.corpus import stopwords
import nltk
import gensim
import re

nltk.download('stopwords')
stop_words = stopwords.words('russian')


def preprocess(text, join_back=True):
    text =  re.sub(r'\n', '', text)
    text = re.sub(r'[^\w\s]','', text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = text.lower()

    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stop_words:
            result.append(token)
    if join_back:
        result = " ".join(result)
    return result



preprocessor_svm = lambda text: preprocess(text)
preprocessor_bert = lambda text: preprocess(text)
