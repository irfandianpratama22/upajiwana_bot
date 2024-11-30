import numpy as np
import json
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Unduh data NLTK untuk tokenisasi (hanya perlu dijalankan sekali)
nltk.download("punkt")

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi tokenisasi
def tokenize(sentence):
    tokens = nltk.word_tokenize(sentence)
    return tokens

# Fungsi stemming
def stem(word):
    stemmed_word = stemmer.stem(word.lower())
    return stemmed_word

# Fungsi bag of words
def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    
    # Tandai indeks sebagai 1 jika kata ditemukan
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    
    return bag
