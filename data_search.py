import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.title("🔍 Semantic Search with Sentence Transformers")

# Sidebar options
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of top results", 1, 20, 5)

# Upload or use default corpus
st.sidebar.subheader("Upload Corpus File (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
else:
    # Sample corpus
    df = pd.DataFrame({
        "text": [
            "Machine learning is a subset of AI.",
            "Streamlit helps build data apps quickly.",
            "Transformers are powerful NLP models.",
            "Semantic similarity is useful for search.",
            "The sun rises in the east."
        ]
    })
    st.info("Using default sample corpus.")

# Ensure 'text' column exists
if "text" not in df.columns:
    st.error("CSV must contain a 'text' column.")
else:
    corpus = df["text"].tolist()
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

    query = st.text_input("Enter your search query:")
    if st.button("Search") and query:
        query_embedding = model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]

        st.subheader("🔎 Search Results")
        for idx, hit in enumerate(hits):
            result_text = corpus[hit['corpus_id']]
            score = hit['score']
            st.markdown(f"**{idx+1}.** {result_text}")
            st.caption(f"Similarity Score: {score:.4f}")
