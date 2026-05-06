# LOTOFACIL IA v0.5.1

import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")

uploaded_file = st.file_uploader("Envie Lotofacil.xlsx", type=["xlsx"])

if uploaded_file is None:
    st.warning("Envie o arquivo para continuar")
    st.stop()

df = pd.read_excel(uploaded_file)
bolas = df[[f"Bola{i}" for i in range(1,16)]]

primos = {2,3,5,7,11,13,17,19,23}
moldura = {1,2,3,4,5,6,10,11,15,16,20,21,22,23,24,25}

def features_jogo(jogo):
    return {
        "soma": sum(jogo),
        "pares": sum(1 for n in jogo if n%2==0),
        "primos": sum(1 for n in jogo if n in primos),
        "mult3": sum(1 for n in jogo if n%3==0),
        "moldura": sum(1 for n in jogo if n in moldura)
    }

def matriz():
    m = pd.DataFrame(0, index=bolas.index, columns=range(1,26))
    for i,row in bolas.iterrows():
        for n in row:
            m.at[i,n]=1
    return m

mat = matriz()

@st.cache_resource
def treinar():
    modelos={}
    X = mat.shift(1).fillna(0)
    for n in range(1,26):
        model = XGBClassifier(eval_metric='logloss')
        model.fit(X, mat[n])
        modelos[n]=model
    return modelos

def probabilidades(modelos):
    ult = mat.iloc[-1].values.reshape(1,-1)
    return {n:modelos[n].predict_proba(ult)[0][1] for n in range(1,26)}

ultimo = set(bolas.iloc[-1].values)

def score(jogo, probs):
    base = sum(probs[n] for n in jogo)
    f = features_jogo(jogo)
    bonus = 0
    if 180 <= f["soma"] <= 220:
        bonus += 1
    if 6 <= f["pares"] <= 9:
        bonus += 1
    repet = len(set(jogo) & ultimo)
    if 8 <= repet <= 10:
        bonus += 2
    return base + bonus

historico = set(tuple(sorted(r)) for r in bolas.values)

def gerar(qtd, probs):
    jogos=[]
    usados=set()
    while len(jogos)<qtd:
        jogo = tuple(sorted(random.sample(range(1,26),15)))
        if jogo in historico or jogo in usados:
            continue
        f = features_jogo(jogo)
        if not (180 <= f["soma"] <= 220):
            continue
        jogos.append((jogo, score(jogo, probs)))
        usados.add(jogo)
    jogos.sort(key=lambda x: x[1], reverse=True)
    return jogos

def clusterizar(jogos, qtd_final, k=10):
    X = [[1 if i in j else 0 for i in range(1,26)] for j,_ in jogos]
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    clusters={}
    for i,(j,sc) in enumerate(jogos):
        clusters.setdefault(labels[i], []).append((j,sc))

    for c in clusters:
        clusters[c].sort(key=lambda x: x[1], reverse=True)

    finais=[]
    while len(finais)<qtd_final:
        for c in clusters:
            if clusters[c]:
                finais.append(clusters[c].pop(0))
            if len(finais)>=qtd_final:
                break

    return finais[:qtd_final]

def heatmap(probs):
    grid = np.zeros((5,5))
    for i,n in enumerate(range(1,26)):
        grid[i//5][i%5] = probs[n]
    return px.imshow(grid, text_auto=True)

st.title("Lotofacil IA v0.5.1")

qtd = st.slider("Quantidade de jogos",1,100,10)

if st.button("Gerar Jogos"):
    modelos = treinar()
    probs = probabilidades(modelos)

    st.plotly_chart(heatmap(probs))

    with st.spinner("Gerando combinações..."):
        jogos = gerar(qtd*5, probs)
        jogos = clusterizar(jogos, qtd_final=qtd, k=10)

    dfj = pd.DataFrame({
        "Jogo":[i+1 for i in range(len(jogos))],
        "Numeros":[list(j) for j,_ in jogos],
        "Score":[round(s,3) for _,s in jogos]
    })

    st.dataframe(dfj, use_container_width=True)
