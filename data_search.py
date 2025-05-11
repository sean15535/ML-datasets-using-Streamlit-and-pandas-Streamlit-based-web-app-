import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load the SentenceTransformer model
@st.cache_resource
def load_model():
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# Cache the embeddings calculation
@st.cache_data
def compute_corpus_embeddings(corpus, _model):
    """Compute and cache embeddings for the given corpus."""
    return _model.encode(corpus, convert_to_tensor=True)

# Main App
def main():
    st.title("🔍 Semantic Search with Sentence Transformers")
    
    # Load model
    model = load_model()

    # Sidebar settings
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Number of top results", 1, 20, 5)

    # Upload or use default corpus
    st.sidebar.subheader("Upload Corpus File (CSV)")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
    else:
        # Default corpus
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

    # Validate corpus
    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column.")
        return

    # Preview corpus
    st.subheader("📄 Corpus Preview")
    st.dataframe(df.head())

    # Compute corpus embeddings
    corpus = df["text"].tolist()
    corpus_embeddings = compute_corpus_embeddings(corpus, model)

    # Search query input
    query = st.text_input("Enter your search query:")
    if st.button("Search") and query:
        # Compute query embedding
        query_embedding = model.encode(query, convert_to_tensor=True)

        # Perform semantic search
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]

        # Display results
        st.subheader("🔎 Search Results")
        results = [
            {"Rank": idx + 1, "Text": corpus[hit['corpus_id']], "Score": hit['score']}
            for idx, hit in enumerate(hits)
        ]
        st.table(results)

# Run the app
if __name__ == "__main__":
    main()