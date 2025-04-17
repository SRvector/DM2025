#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App Launcher

To deploy this Streamlit app, simply run:
    streamlit run app.py
"""
import pandas as pd
import re
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import plotly.express as px

# Page config must come before other Streamlit calls
st.set_page_config(
    page_title="Apriori & Clustering Dashboard",
    layout="wide"
)

# -------------------- Preprocessing --------------------
def clean_text(text):
    """
    Basic text cleaning: lowercase, remove URLs and non-alphanumeric chars.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.[^\s]+', '', text)
    text = re.sub(r'[^\w\s#]', '', text)
    return text

@st.cache_data
def load_data(csv_file):
    """
    Load CSV and prepare text and hashtags.
    """
    df = pd.read_csv(csv_file, encoding='latin1')
    # Choose text column
    text_col = 'original_text' if 'original_text' in df.columns else 'text'
    df['clean_text'] = df[text_col].fillna('').apply(clean_text)
    # Ensure hashtags column exists
    if 'hashtags' not in df.columns:
        df['hashtags'] = df['clean_text'].str.findall(r"#(\w+)").apply(lambda lst: ' '.join(lst))
    return df

@st.cache_data
def extract_transactions(df):
    """
    Convert hashtags column into list of tags per tweet.
    """
    return df['hashtags'].fillna('').apply(lambda x: x.split()).tolist()

@st.cache_data
def onehot_encode(transactions):
    """
    Convert list of tag lists into one-hot DataFrame.
    """
    items = sorted({item for trans in transactions for item in trans})
    return pd.DataFrame([{item: (item in trans) for item in items} for trans in transactions])

@st.cache_data
def vectorize_text(text_series):
    """
    TF-IDF vectorization.
    """
    vect = TfidfVectorizer(stop_words='english', max_features=1000)
    return vect, vect.fit_transform(text_series)

# -------------------- Main App --------------------
def main():
    st.title("Interactive Apriori & Clustering Dashboard")

    # Sidebar: file upload
    csv_uploader = st.sidebar.file_uploader("Upload CSV dataset", type="csv")
    if not csv_uploader:
        st.sidebar.info("Please upload a CSV file containing your dataset.")
        return

    # Load & preview data
    df = load_data(csv_uploader)
    st.write("### Data Preview")
    st.dataframe(df.head(3))

    # Prepare transactions & one-hot
    transactions = extract_transactions(df)
    oht = onehot_encode(transactions)

    # Sidebar: Apriori settings
    st.sidebar.header("Apriori Settings")
    min_support = st.sidebar.slider("Min Support", 0.01, 0.2, 0.02, 0.01)
    min_conf = st.sidebar.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)
    min_lift = st.sidebar.slider("Min Lift", 1.0, 10.0, 1.2, 0.2)

    # Sidebar: Clustering settings
    st.sidebar.header("Clustering Settings")
    algo = st.sidebar.selectbox("Algorithm", ["KMeans", "DBSCAN"])

    # Compute Apriori rules
    freq_itemsets = apriori(oht, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_conf)
    rules = rules[rules['lift'] >= min_lift]

    # Vectorize text for clustering
    vect, X = vectorize_text(df['clean_text'])
    if algo == 'KMeans':
        n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 10, 4)
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X)
    else:
        eps = st.sidebar.slider("DBSCAN eps", 0.1, 1.0, 0.5, 0.1)
        min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 10, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
    df['cluster'] = labels

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Association Rules", "Clustering", "Dashboard"])

    # Tab1: Association Rules
    with tab1:
        st.subheader("Frequent Itemsets")
        st.dataframe(freq_itemsets)
        st.subheader("Filtered Rules")
        st.dataframe(rules)
        if not rules.empty:
            fig = px.scatter(
                rules, x='support', y='confidence', size='lift', color='lift',
                hover_data=['antecedents', 'consequents'],
                title="Support vs Confidence (size=Lift)"
            )
            st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            label="Download Rules CSV", data=rules.to_csv(index=False),
            file_name="association_rules.csv", mime="text/csv"
        )

    # Tab2: Clustering
    with tab2:
        coords = PCA(n_components=2, random_state=42).fit_transform(X.toarray())
        fig2 = px.scatter(
            x=coords[:, 0], y=coords[:, 1], color=df['cluster'].astype(str),
            title=f"{algo} Clusters (2D PCA)", hover_data=['clean_text']
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.subheader("Sample Cluster Assignments")
        st.dataframe(df[['clean_text', 'cluster']].head())

    # Tab3: Dashboard
    with tab3:
        st.header("Project Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Tweets", len(df))
        c2.metric("Unique Hashtags", len(oht.columns))
        c3.metric("Itemsets", len(freq_itemsets))
        c4.metric("Rules", len(rules))

        if not rules.empty:
            fig3 = px.histogram(rules, x='support', nbins=20, title="Rule Support Distribution")
            st.plotly_chart(fig3, use_container_width=True)

        cluster_counts = df['cluster'].value_counts().reset_index()
        cluster_counts.columns = ['cluster', 'count']
        fig4 = px.bar(cluster_counts, x='cluster', y='count', title="Cluster Sizes")
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Explore Tweets by Cluster")
        sel = st.selectbox("Select Cluster", sorted(cluster_counts['cluster'].tolist()))
        for i, tweet in enumerate(
            df[df['cluster'] == sel]['clean_text'].sample(min(5, len(df[df['cluster'] == sel]))), 1
        ):
            st.write(f"{i}. {tweet}")

if __name__ == "__main__":
    main()
