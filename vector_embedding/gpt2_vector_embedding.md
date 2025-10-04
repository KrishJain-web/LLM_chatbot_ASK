!pip install gensim
import gensim.downloader as api
model = api.load("word2vec-google-news-300")
word_vectors = model
print(word_vectors['computer']) #300 vectors, (word2vec trained model)


<img width="752" height="693" alt="image" src="https://github.com/user-attachments/assets/03c45018-9862-4d55-adce-19df7afc7c8f" />

