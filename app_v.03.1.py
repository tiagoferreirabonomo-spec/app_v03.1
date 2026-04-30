# =========================================================
# LOTOFACIL IA HIBRIDA - v0.3.3 (100 COMBINACOES OTIMIZADO)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
from xgboost import XGBClassifier

st.set_page_config(layout="wide")

# =============================
# UPLOAD DO ARQUIVO
# =============================
uploaded_file = st.file_uploader("Envie o arquivo Lotofacil.xlsx", type=["xlsx"])

if uploaded_file is None:
    st.warning("Envie o arquivo para continuar")
    st.stop()

# =============================
# DADOS
# =============================
@st.cache_data
def carregar(uploaded_file):
    df = pd.read_excel(uploaded_file)
    bolas = df[[f"Bola{i}" for i in range(1, 16)]]
    return df, bolas

df, bolas = carregar(uploaded_file)

# =============================
# MATRIZ BINARIA
# =============================
def matriz_binaria():
    m = pd.DataFrame(0, index=bolas.index, columns=range(1,26))
    for i,row in bolas.iterrows():
        for n in row:
            m.at[i,n]=1
    return m

matriz = matriz_binaria()

# =============================
# MODELO
# =============================
@st.cache_resource
def treinar():
    modelos={}
    X = matriz.shift(1).fillna(0)
    for n in range(1,26):
        model = XGBClassifier(eval_metric='logloss')
        model.fit(X, matriz[n])
        modelos[n]=model
    return modelos

def probabilidades(modelos):
    ultimo = matriz.iloc[-1].values.reshape(1,-1)
    return {n:modelos[n].predict_proba(ultimo)[0][1] for n in range(1,26)}

# =============================
# HISTORICO
# =============================
historico = set(tuple(sorted(row)) for row in bolas.values)

# =============================
# SCORE
# =============================
def score(jogo, probs):
    return sum(probs[n] for n in jogo)

# =============================
# GENETICO
# =============================
def crossover(p1, p2):
    corte = random.randint(5,10)
    filho = list(set(p1[:corte] + p2[corte:]))

    while len(filho) < 15:
        filho.append(random.randint(1,25))
        filho = list(set(filho))

    return tuple(sorted(filho[:15]))

def mutacao(jogo):
    jogo = list(jogo)
    jogo[random.randint(0,14)] = random.randint(1,25)
    return tuple(sorted(set(jogo)))[:15]

def evoluir(pop):
    nova=[]
    for _ in range(len(pop)):
        p1,p2 = random.sample(pop,2)
        filho = crossover(p1,p2)
        if random.random()<0.3:
            filho = mutacao(filho)
        nova.append(filho)
    return nova

# =============================
# MONTE CARLO OTIMIZADO
# =============================
def monte_carlo(jogo, probs, n=10):
    return sum(score(jogo,probs) for _ in range(n))/n

# =============================
# GERADOR AVANCADO (100 JOGOS)
# =============================
def gerar_avancado(qtd, probs):

    pop = [tuple(sorted(random.sample(range(1,26),15))) for _ in range(3000)]

    for _ in range(4):
        pop = evoluir(pop)

    avaliados = [(j, monte_carlo(j,probs)) for j in pop if j not in historico]
    avaliados.sort(key=lambda x: x[1], reverse=True)

    finais=[]

    for jogo,sc in avaliados:
        if all(len(set(jogo)&set(j2)) < 13 for j2,_ in finais):
            finais.append((jogo,sc))

        if len(finais) >= qtd:
            break

    usados = set(j for j,_ in finais)

    while len(finais) < qtd:
        novo = tuple(sorted(random.sample(range(1,26),15)))
        if novo not in historico and novo not in usados:
            finais.append((novo, score(novo, probs)))
            usados.add(novo)

    return finais

# =============================
# UI
# =============================
st.title("Lotofacil IA v0.3.3 - 100 Combinações")

if st.checkbox("Mostrar dados carregados"):
    st.dataframe(df.head())

qtd = st.slider("Quantidade de jogos", 1, 100, 10)

if st.button("Gerar IA Avancada"):

    modelos = treinar()
    probs = probabilidades(modelos)

    dfp = pd.DataFrame({
        "Numero": list(probs.keys()),
        "Probabilidade": list(probs.values())
    })

    st.plotly_chart(px.bar(dfp, x="Numero", y="Probabilidade",
                           title="Probabilidade dos numeros"))

    with st.spinner("Gerando combinações..."):
        jogos = gerar_avancado(qtd, probs)

    df_jogos = pd.DataFrame({
        "Jogo": [i+1 for i in range(len(jogos))],
        "Numeros": [list(j) for j,_ in jogos],
        "Score": [round(sc,4) for _,sc in jogos]
    })

    st.dataframe(df_jogos, use_container_width=True)
