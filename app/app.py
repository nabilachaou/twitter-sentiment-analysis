import streamlit as st
import pandas as pd
import joblib
import base64
from utils.preprocessing import clean_text
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

# --- CSS personnalisé ---
def load_custom_css():
    st.markdown("""<style> ... [TON CSS ACTUEL ICI] ... </style>""", unsafe_allow_html=True)

# --- Ajouter fond d'écran ---
def add_background(image_path):
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        ext = "png" if image_path.lower().endswith(".png") else "jpg"
        img_base64 = base64.b64encode(data).decode()
        img_url = f"data:image/{ext};base64,{img_base64}"
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{img_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            filter: brightness(0.85);
        }}
        </style>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Image de fond non trouvée.")

# --- Chargement modèle & vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load("model/final_model.pkl")
        vectorizer = joblib.load("model/vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("Fichiers modèle/vectorizer non trouvés.")
        return None, None

# --- Chargement tweets ---
@st.cache_data(ttl=30)
def load_tweets():
    try:
        df = pd.read_csv("data/stream_tweets.csv")
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        return df.dropna(subset=['date'])
    except FileNotFoundError:
        st.error("Fichier stream_tweets.csv introuvable.")
        return pd.DataFrame()

# --- Graphique barres ---
def create_sentiment_chart(sentiment_counts):
    colors = {'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#f59e0b'}
    fig = go.Figure(data=[go.Bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        marker_color=[colors.get(s, '#667eea') for s in sentiment_counts.index],
        text=sentiment_counts.values,
        textposition='auto'
    )])
    fig.update_layout(template="plotly_dark", height=400)
    return fig

# --- Graphique pie ---
def create_sentiment_pie_chart(sentiment_counts):
    colors = ['#10b981', '#ef4444', '#f59e0b']
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=.3,
        marker_colors=colors[:len(sentiment_counts)]
    )])
    fig.update_layout(template="plotly_dark", height=400)
    return fig

# --- Graphique temporel ---
def create_timeline_chart(df):
    df_time = df.set_index('date').resample('1T')['sentiment_pred'].value_counts().unstack(fill_value=0)
    fig = go.Figure()
    colors = {'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#f59e0b'}
    for sentiment in df_time.columns:
        fig.add_trace(go.Scatter(
            x=df_time.index,
            y=df_time[sentiment],
            mode='lines+markers',
            name=sentiment.title(),
            line=dict(color=colors.get(sentiment, '#667eea'))
        ))
    fig.update_layout(template="plotly_dark", height=400)
    return fig

# --- Interface principale ---
def main():
    st.set_page_config(page_title="Dashboard Sentiment", layout="wide")
    load_custom_css()

    image_path = "C:/Users/dell/Desktop/my project/image/fond_ecran.jpg"
    add_background(image_path)

    model, vectorizer = load_model_and_vectorizer()
    if model is None or vectorizer is None:
        st.stop()

    with st.container():
        st.markdown('<h1 class="main-title">Dashboard d\'Analyse de Sentiment en Temps Réel</h1>', unsafe_allow_html=True)

        with st.sidebar:
            st.markdown('<div class="sidebar-container">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-title">Paramètres</div>', unsafe_allow_html=True)
            n_tweets_display = st.slider("Tweets à afficher", 10, 1000, 100, 10)
            refresh_interval = st.number_input("Rafraîchissement (secondes)", 5, 300, 30, 5)
            auto_refresh = st.checkbox("Rafraîchissement automatique", True)
            st.markdown('</div>', unsafe_allow_html=True)

        df = load_tweets()
        if df.empty:
            st.stop()

        df["clean_text"] = df["text"].apply(clean_text)
        X = vectorizer.transform(df["clean_text"])
        df["sentiment_pred"] = model.predict(X)

        sentiment_counts = df["sentiment_pred"].value_counts()
        total = len(df)

        st.subheader("Statistiques Globales")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tweets", total)
        col2.metric("Positifs", sentiment_counts.get("positive", 0))
        col3.metric("Négatifs", sentiment_counts.get("negative", 0))
        col4.metric("Neutres", sentiment_counts.get("neutral", 0))

        st.subheader("Visualisations")
        col1, col2 = st.columns(2)
        col1.plotly_chart(create_sentiment_chart(sentiment_counts), use_container_width=True)
        col2.plotly_chart(create_sentiment_pie_chart(sentiment_counts), use_container_width=True)

        if len(df) > 1:
            st.subheader("Évolution Temporelle")
            st.plotly_chart(create_timeline_chart(df), use_container_width=True)

        st.subheader("Tweets Récents")
        df_display = df.sort_values("date", ascending=False).head(n_tweets_display).copy()
        df_display["sentiment_pred"] = df_display["sentiment_pred"].apply(
            lambda x: f"{'🟢' if x=='positive' else '🔴' if x=='negative' else '🟡'} {x.title()}"
        )
        df_display["date"] = df_display["date"].dt.strftime("%d/%m/%Y %H:%M:%S")

        st.dataframe(df_display[["date", "user", "text", "sentiment_pred"]].rename(columns={
            "date": "Date", "user": "Utilisateur", "text": "Tweet", "sentiment_pred": "Sentiment"
        }), height=400, use_container_width=True)

        if auto_refresh:
            st_autorefresh(interval=refresh_interval * 1000, limit=None, key="autorefresh")

# --- Lancement ---
if __name__ == "__main__":
    main()