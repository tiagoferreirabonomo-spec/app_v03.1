# =========================================================
# LOTOFACIL IA HIBRIDA - v0.3.1 (ESTAVEL E OTIMIZADA)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
from xgboost import XGBClassifier

st.set_page_config(layout="wide")

@st.cache_data
def carregar():
    df = pd.read_excel("Lotofacil.xlsx")
    bolas = df[[f"Bola{i}" for i in range(1, 16)]]
    return df, bolas

df, bolas = carregar()

def matriz_binaria():
    m = pd.DataFrame(0, index=bolas.index, columns=range(1,26))
    for i,row in bolas.iterrows():
        for n in row:
            m.at[i,n]=1
    return m

matriz = matriz_binaria()

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

historico = set(tuple(sorted(row)) for row in bolas.values)

def score(jogo, probs):
    return sum(probs[n] for n in jogo)

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

def monte_carlo(jogo, probs, n=30):
    return sum(score(jogo,probs) for _ in range(n))/n

def gerar_avancado(qtd, probs):

    pop = [tuple(sorted(random.sample(range(1,26),15))) for _ in range(1000)]

    for _ in range(6):
        pop = evoluir(pop)

    avaliados = [(j, monte_carlo(j,probs)) for j in pop if j not in historico]
    avaliados.sort(key=lambda x: x[1], reverse=True)

    finais=[]

    for jogo,sc in avaliados:
        if all(len(set(jogo)&set(j2)) < 12 for j2,_ in finais):
            finais.append((jogo,sc))

        if len(finais) >= qtd:
            break

    if len(finais) < qtd:
        faltantes = qtd - len(finais)
        usados = set(j for j,_ in finais)

        extras = [(j,sc) for j,sc in avaliados if j not in usados][:faltantes]
        finais.extend(extras)

    return finais

st.title("Lotofacil IA v0.3.1 - Estavel")

qtd = st.slider("Quantidade de jogos", 1, 20, 5)

if st.button("Gerar IA Avancada"):

    modelos = treinar()
    probs = probabilidades(modelos)

    dfp = pd.DataFrame({
        "Numero": list(probs.keys()),
        "Probabilidade": list(probs.values())
    })

    st.plotly_chart(px.bar(dfp, x="Numero", y="Probabilidade",
                           title="Probabilidade dos numeros"))

    jogos = gerar_avancado(qtd, probs)

    if len(jogos) < qtd:
        st.warning(f"Foram gerados apenas {len(jogos)} jogos validos.")

    st.subheader("Jogos Gerados")

    for i,(j,sc) in enumerate(jogos,1):
        st.success(f"Jogo {i}: {list(j)} | Score: {round(sc,4)}")
