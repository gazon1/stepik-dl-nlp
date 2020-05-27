import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_20newsgroups
from dlnlputils.data import tokenize_corpus, build_vocabulary, vectorize_texts, tokenize_text_simple_regex
from dlnlputils.pipeline import init_random_seed
import pickle


init_random_seed()

test_source = fetch_20newsgroups(subset='test')

# train_source_aug = pickle.load(open('train_source_aug.pkl', 'rb'))
# train_target_aug = pickle.load(open('train_target_aug.pkl', 'rb'))
train_source = fetch_20newsgroups(subset='train')
train_source_aug = train_source['data']
train_target_aug = train_source['target']

print(len(train_source_aug))
# print(len(train_target_aug))

train_tokenized = tokenize_corpus(train_source_aug, tokenize_text_simple_regex, ngram_range=(1,2))
test_tokenized = tokenize_corpus(test_source['data'], tokenize_text_simple_regex, ngram_range=(1,2))

print(len(train_tokenized))
# print(len(test_tokenized))

import nltk
#nltk.download('wordnet')
lemma = nltk.wordnet.WordNetLemmatizer()
sno = nltk.stem.SnowballStemmer('english')

def preprocces_word(word):
    # lemmatized = lemma.lemmatize(word)
    # return sno.stem(lemmatized)
    return lemma.lemmatize(word)

# Нормализация

train_tokenized = [[preprocces_word(word) for word in sentence] for sentence in train_tokenized]
test_tokenized = [[preprocces_word(word) for word in sentence] for sentence in test_tokenized]

# print(len(train_tokenized_lemmatized))


# подсчет tf-idf

MAX_DF = 0.8
MIN_COUNT = 5
vocabulary, word_doc_freq = build_vocabulary(train_tokenized, max_doc_freq=MAX_DF, min_count=MIN_COUNT)
UNIQUE_WORDS_N = len(vocabulary)
print('Количество уникальных токенов', UNIQUE_WORDS_N)
with open('UNIQUE_WORDS_N.txt', 'w') as f:
    print(UNIQUE_WORDS_N, file=f)
print(list(vocabulary.items())[:10])

# Векторизация текстов

VECTORIZATION_MODE = 'pmi'
train_vectors = vectorize_texts(train_tokenized, vocabulary, word_doc_freq, mode=VECTORIZATION_MODE)
test_vectors = vectorize_texts(test_tokenized, vocabulary, word_doc_freq, mode=VECTORIZATION_MODE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)
scaler.fit_transform(train_vectors)
scaler.transform(test_vectors)

print('Размерность матрицы признаков обучающей выборки', train_vectors.shape)
print('Размерность матрицы признаков тестовой выборки', train_vectors.shape)
print()
print('Количество ненулевых элементов в обучающей выборке', train_vectors.nnz)
print('Процент заполненности матрицы признаков {:.2f}%'.format(train_vectors.nnz * 100 / (train_vectors.shape[0] * train_vectors.shape[1])))
print()
print('Количество ненулевых элементов в тестовой выборке', test_vectors.nnz)
print('Процент заполненности матрицы признаков {:.2f}%'.format(test_vectors.nnz * 100 / (test_vectors.shape[0] * test_vectors.shape[1])))

pickle.dump(train_vectors, open('train_vectors.pkl', 'wb'))
pickle.dump(test_vectors, open('test_vectors.pkl', 'wb'))