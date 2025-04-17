import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import zipfile

st.title("Apriori and Clustering Analysis")

uploaded_file = st.file_uploader("Upload ZIP file containing your CSV dataset", type="zip")

if uploaded_file:
    try:
        with zipfile.ZipFile(uploaded_file) as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                st.error("No CSV file found in the ZIP.")
            else:
                data = pd.read_csv(z.open(csv_files[0]))
                st.write("### Dataset Preview", data.head())

                df_encoded = pd.get_dummies(data)
                st.write("### One-Hot Encoded Data", df_encoded.head())

                min_support = st.sidebar.slider("Minimum Support", 0.01, 0.5, 0.05)
                apriori_items = apriori(df_encoded, min_support=min_support, use_colnames=True)
                st.write("### Frequent Itemsets", apriori_items)

                min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.5)
                rules = association_rules(apriori_items, metric="confidence", min_threshold=min_confidence)
                st.write("### Association Rules", rules)

                n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
                pca_n = st.sidebar.slider("PCA Components", 2, min(10, df_encoded.shape[1]), 2)
                pca = PCA(n_components=pca_n)
                components = pca.fit_transform(df_encoded)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(components)
                data['Cluster'] = clusters
                st.write("### Clustered Data", data.head())

                fig, ax = plt.subplots()
                scatter = ax.scatter(components[:, 0], components[:, 1], c=clusters)
                ax.set_title("Cluster Plot (PCA-reduced)")
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Error processing file: {e}")
