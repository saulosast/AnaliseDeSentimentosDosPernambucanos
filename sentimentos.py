# Imports necessários
import pandas as pd
import random
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px

# Baixar recursos do NLTK
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# Gerar dados simulados
def gerar_dados():
    cidades = ['Recife', 'Olinda', 'Caruaru', 'Ipojuca', 'Garanhuns', 'Petrolina', 'Gravatá', 'Jaboatão', 'Serra Talhada', 'Arcoverde']
    positivos = ["Adorei o atendimento", "Comida deliciosa", "Excelente serviço", "Muito bom", "Gostei bastante", "Ambiente acolhedor", "Produtos bons", "Preço justo", "Voltarei com certeza", "Experiência muito positiva"]
    neutros = ["Nada de mais", "Ok", "Foi tudo normal", "Atendimento razoável", "Serviço dentro do esperado", "Achei comum", "Loja simples", "Ambiente ok", "Nada que chame atenção", "Tudo certo"]
    negativos = ["Muito ruim", "Atendimento fraco", "Comida horrível", "Não gostei", "Preço muito alto", "Experiência ruim", "Demora no atendimento", "Falta organização", "Ambiente sujo", "Não volto mais"]

    random.seed(42)
    dados = []
    for _ in range(1000):
        cidade = random.choice(cidades)
        sentimento = random.choices(['positivo', 'neutro', 'negativo'], weights=[0.5, 0.3, 0.2])[0]
        if sentimento == 'positivo':
            comentario = random.choice(positivos)
        elif sentimento == 'neutro':
            comentario = random.choice(neutros)
        else:
            comentario = random.choice(negativos)
        dados.append({'cidade': cidade, 'comentario': comentario, 'sentimento': sentimento})
    return pd.DataFrame(dados)

# Limpeza de texto
stop_pt = set(stopwords.words('portuguese'))
def limpar_texto(txt):
    txt = txt.lower()
    txt = re.sub(f"[{string.punctuation}]", "", txt)
    tokens = word_tokenize(txt)
    tokens = [w for w in tokens if w not in stop_pt]
    return " ".join(tokens)

# Dados
df = gerar_dados()
df['comentario_limpo'] = df['comentario'].apply(limpar_texto)

# Vetorização e modelo
vetor = TfidfVectorizer()
X = vetor.fit_transform(df['comentario_limpo'])
y = df['sentimento']

# Divisão treino/teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo Random Forest
modelo = RandomForestClassifier(n_estimators=100)
modelo.fit(X_treino, y_treino)
df['sentimento_previsto'] = modelo.predict(X)

# Resultados de classificação
print("Relatório de Classificação do Modelo:")
print(classification_report(y_teste, df['sentimento_previsto'][y_teste.index]))

# Visualizações
# 1. Distribuição geral de sentimentos
fig1 = px.histogram(df, x='sentimento_previsto', color='sentimento_previsto', title='Distribuição Geral de Sentimentos', text_auto=True)
fig1.show()

# 2. Sentimento por cidade
fig2 = px.histogram(df, x='cidade', color='sentimento_previsto', barmode='group', title='Sentimentos por Cidade', text_auto=True)
fig2.update_layout(xaxis_title='Cidade', yaxis_title='Volume de Comentários')
fig2.show()

# 3. Mapa de calor de sentimentos por cidade
