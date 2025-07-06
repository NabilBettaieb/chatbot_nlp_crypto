
import streamlit as st
import pandas as pd
from datetime import datetime
import re
import matplotlib.pyplot as plt
from difflib import get_close_matches

@st.cache_data
def charger_donnees():
    df = pd.read_csv("tweet_analysis.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["crypto"] = df["cryptos"].str.replace(r'\$', '', regex=True).str.upper()
    df["sentiment_numeric"] = df["sentiment"].map({"negative": -1, "neutral": 0, "positive": 1})
    return df

df = charger_donnees()
crypto_list = sorted(df["crypto"].unique().tolist())

# Synonymes simples manuels
synonymes = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "eth": "ETH",
    "ethereum": "ETH",
    "bnb": "BNB",
    "binance": "BNB",
    "op": "OP",
    "optimism": "OP",
    "matic": "MATIC",
    "polygon": "MATIC"
}

def detecter_crypto_avec_fautes(mot):
    mot_clean = re.sub(r"[^a-zA-Z]", "", mot.lower())
    if mot_clean in synonymes:
        return synonymes[mot_clean]
    matches = get_close_matches(mot_clean.upper(), crypto_list, n=1, cutoff=0.8)
    return matches[0] if matches else None

def extraire_infos(question):
    found_cryptos = []
    mots = question.lower().split()
    for mot in mots:
        c = detecter_crypto_avec_fautes(mot)
        if c and c not in found_cryptos:
            found_cryptos.append(c)

    match_intervalle = re.search(r"(\d{1,2})h\s*(et|-|à)\s*(\d{1,2})h", question)
    if match_intervalle:
        start_hour = int(match_intervalle.group(1))
        end_hour = int(match_intervalle.group(3))
    else:
        match_h = re.search(r"(\d{1,2})h", question)
        if match_h:
            start_hour = int(match_h.group(1))
            end_hour = start_hour
        else:
            start_hour, end_hour = 9, 12

    return list(set(found_cryptos)), start_hour, end_hour

def analyse_crypto(crypto, start_hour, end_hour):
    df_filtered = df[
        (df["crypto"] == crypto) &
        (df["Date"].dt.date == datetime(2022, 11, 30).date()) &
        (df["Date"].dt.hour >= start_hour) &
        (df["Date"].dt.hour <= end_hour)
    ]
    if df_filtered.empty:
        return None
    sentiment = df_filtered["sentiment"].mode()[0]
    score = round(df_filtered["score"].mean(), 3)
    volume = len(df_filtered)
    sentiment_counts = df_filtered["sentiment"].value_counts().to_dict()
    return {"crypto": crypto, "sentiment": sentiment, "score": score, "volume": volume, "counts": sentiment_counts}

def classement_cryptos(start_hour, end_hour, top_n=None, worst=False):
    df_filtered = df[
        (df["Date"].dt.date == datetime(2022, 11, 30).date()) &
        (df["Date"].dt.hour >= start_hour) &
        (df["Date"].dt.hour <= end_hour)
    ]
    moyenne = df_filtered.groupby("crypto")["score"].mean()
    moyenne = moyenne.sort_values(ascending=worst)
    return moyenne.head(top_n) if top_n else moyenne

st.set_page_config(page_title="Chatbot Crypto", layout="wide")
st.title("🤖 Chatbot Crypto – Tweets du 30/11/2022")

with st.expander("🧭 Guide d'utilisation"):
    st.markdown("""
    Ce chatbot utilise des tweets du **30/11/2022 entre 9h et 12h** pour répondre à vos questions sur les cryptomonnaies.

    ### 💬 Exemples de questions que vous pouvez poser :
    #### Analyse d’une crypto :
    - Que pense-t-on de **BTC** ?
    - Score moyen **ETH**

    #### Comparaison entre deux cryptos :
    - Qui est mieux entre **ETH et BTC** ?
    - Est-ce que **ETH** est mieux que **BNB** ?

    #### Classement :
    - Top 3 cryptos les mieux notées
    - Quelles sont les cryptos les moins bien notées à 10h ?

    #### Visualisation :
    - Montre l'évolution de **BTC**
    - Affiche un graphique de **ETH**

    ⚠️ Les noms des cryptos peuvent être écrits avec ou sans fautes : `ethereum`, `eth`, `bitcoin`, `btc`, etc.
    """)

with st.expander("🧑‍💻 Êtes-vous développeur ?"):
    st.markdown("""
    ### 🔌 Utiliser le chatbot comme une API (localement)

    Ce projet est actuellement en Streamlit. Pour l'utiliser en tant qu'API :

    - Clonez ce projet et lancez le chatbot :
    ```bash
    streamlit run chatbot_app.py
    ```
    - Ou intégrez la fonction `analyse_crypto()` dans une API Flask ou FastAPI.

    Exposer le modèle :
    - `analyse_crypto('BTC', 9, 12)` → renvoie score moyen, sentiment dominant, volume, etc.
    """)

tab1, tab2 = st.tabs(["💬 Chatbot", "📊 Visualisations"])

with tab1:
    question = st.text_input("💬 Pose ta question ici :", key="question_input")

    if st.button("Analyser", key="analyser_btn") and question:
        q = question.lower()
        cryptos, start_hour, end_hour = extraire_infos(q)

        if "top 3" in q or "meilleures cryptos" in q:
            top = classement_cryptos(start_hour, end_hour, top_n=3)
            st.markdown(f"🏆 **Top 3 cryptos** entre {start_hour}h et {end_hour}h :")
            for i, (c, s) in enumerate(top.items(), 1):
                st.markdown(f"{i}. **{c}** – score : {round(s, 3)}")
        elif "moins noté" in q or "moins bien" in q or "pire" in q:
            classement = classement_cryptos(start_hour, end_hour, top_n=1, worst=True)
            if not classement.empty:
                worst = classement.index[0]
                score = round(classement.iloc[0], 3)
                st.error(f"⚠️ La crypto la moins bien notée entre {start_hour}h et {end_hour}h est **{worst}** avec un score de **{score}**.")
        elif "mieux noté" in q or "meilleure" in q:
            classement = classement_cryptos(start_hour, end_hour, top_n=1)
            if not classement.empty:
                best = classement.index[0]
                score = round(classement.iloc[0], 3)
                st.success(f"🥇 La crypto la mieux notée entre {start_hour}h et {end_hour}h est **{best}** avec un score de **{score}**.")
        elif "graph" in q or "évolution" in q or "montre" in q or "affiche" in q:
            if len(cryptos) == 1:
                df_crypto = df[
                    (df["crypto"] == cryptos[0]) &
                    (df["Date"].dt.date == datetime(2022, 11, 30).date())
                ].sort_values("Date")
                if df_crypto.empty:
                    st.warning(f"Aucune donnée trouvée pour {cryptos[0]}.")
                else:
                    plt.figure(figsize=(10, 4))
                    plt.plot(df_crypto["Date"], df_crypto["sentiment_numeric"], marker='o')
                    plt.title(f"Évolution binaire du sentiment – {cryptos[0]}")
                    plt.xlabel("Heure")
                    plt.ylabel("Sentiment")
                    plt.yticks([-1, 0, 1], ["Négatif", "Neutre", "Positif"])
                    plt.grid(True)
                    st.pyplot(plt)
            else:
                st.warning("Merci de spécifier **une seule crypto** pour la visualisation.")
        elif len(cryptos) == 1:
            result = analyse_crypto(cryptos[0], start_hour, end_hour)
            if result:
                st.markdown(f"📊 Pour **{result['crypto']}** entre {start_hour}h et {end_hour}h le 30/11/2022 :")
                st.markdown(f"- Sentiment dominant : **{result['sentiment']}**")
                st.markdown(f"- Score moyen : **{result['score']}**")
                st.markdown(f"- Volume de tweets : {result['volume']}")
                st.markdown("**Détail des sentiments :**")
                for k in ["positive", "neutral", "negative"]:
                    count = result["counts"].get(k, 0)
                    st.markdown(f"- {k.capitalize()} : {count} tweet(s)")
            else:
                st.error(f"❌ Aucune donnée disponible pour {cryptos[0]}.")
        elif len(cryptos) == 2:
            r1 = analyse_crypto(cryptos[0], start_hour, end_hour)
            r2 = analyse_crypto(cryptos[1], start_hour, end_hour)
            if r1 and r2:
                st.markdown(f"🔍 Comparaison entre **{r1['crypto']}** et **{r2['crypto']}** entre {start_hour}h et {end_hour}h :")
                st.markdown(f"➡️ {r1['crypto']} : score {r1['score']} ({r1['sentiment']})")
                st.markdown(f"➡️ {r2['crypto']} : score {r2['score']} ({r2['sentiment']})")
                if r1['score'] > r2['score']:
                    st.success(f"✅ **{r1['crypto']}** est mieux perçue.")
                elif r2['score'] > r1['score']:
                    st.success(f"✅ **{r2['crypto']}** est mieux perçue.")
                else:
                    st.info("⚖️ Les deux cryptos sont perçues de manière équivalente.")
            else:
                if not r1:
                    st.error(f"❌ Aucune donnée pour {cryptos[0]}.")
                if not r2:
                    st.error(f"❌ Aucune donnée pour {cryptos[1]}.")
        else:
            st.warning("Je ne comprends pas la question ou aucune crypto détectée.")

with tab2:
    st.subheader("📊 Évolution binaire du sentiment par crypto")
    cryptos_selectionnees = st.multiselect(
        "Choisis les cryptos à afficher (max 5)", options=crypto_list, default=crypto_list[:2]
    )

    if cryptos_selectionnees:
        fig, ax = plt.subplots(figsize=(12, 5))
        for crypto in cryptos_selectionnees:
            df_crypto = df[
                (df["crypto"] == crypto) &
                (df["Date"].dt.date == datetime(2022, 11, 30).date())
            ].sort_values("Date")
            if not df_crypto.empty:
                ax.plot(df_crypto["Date"], df_crypto["sentiment_numeric"], marker='o', label=crypto)
        ax.set_title("Évolution binaire du sentiment (1: Positif, 0: Neutre, -1: Négatif)")
        ax.set_xlabel("Heure")
        ax.set_ylabel("Sentiment")
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["Négatif", "Neutre", "Positif"])
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("Sélectionne au moins une crypto pour afficher son graphique.")
