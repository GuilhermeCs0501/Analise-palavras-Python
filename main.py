import nltk
import string
from unidecode import unidecode
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

texto = """Uma confusão dentro de uma funerária no bairro São Marcos, em Nova Serrana, no Centro-Oeste de Minas Gerais, terminou em agressões, danos materiais e intervenção da Polícia Militar (PM) na manhã de segunda-feira (30). Veja o registro da pancadaria no vídeo acima.
De acordo com a Polícia Militar, a ocorrência foi registrada como 'vias de fato', motivada por um desacordo comercial. A discussão envolveu o dono da funerária e familiares de um homem de 69 anos que morreu horas antes."""

def normalizar(texto):
    texto = texto.lower()
    texto = unidecode(texto)
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    return texto

texto_norm = normalizar(texto)

stop_words = set(stopwords.words('portuguese'))

tokens = texto_norm.split()

tokens_limpos = [t for t in tokens if t not in stop_words]


vectorizer = TfidfVectorizer(stop_words=stopwords.words('portuguese'))

tfidf_matrix = vectorizer.fit_transform([texto_norm])

feature_names = vectorizer.get_feature_names_out()
scores = tfidf_matrix.toarray()[0]

tfidf_scores = list(zip(feature_names, scores))

tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

top_10 = tfidf_scores[:10]

print("\nTop 10 termos:")
for termo, score in top_10:
    print(f"{termo}: {score:.4f}")

frequencias = dict(top_10)

wordcloud = WordCloud(width=800, height=400, background_color='white')
wordcloud.generate_from_frequencies(frequencias)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
wordcloud.to_file("exemplo.png")