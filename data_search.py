import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re

# Load the SentenceTransformer model
@st.cache_resource
def load_model(model_name):
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer(model_name)

# Cache the embeddings calculation
@st.cache_data
def compute_corpus_embeddings(corpus, _model):
    """Compute and cache embeddings for the given corpus."""
    return _model.encode(corpus, convert_to_tensor=True)

# Highlight matched query words in the results
def highlight_query_words(text, query):
    """Highlight query words in the text using a simple regex."""
    query_words = re.escape(query).split()
    pattern = r'|'.join(query_words)
    highlighted_text = re.sub(pattern, lambda match: f"<mark>{match.group(0)}</mark>", text, flags=re.IGNORECASE)
    return highlighted_text

# Filter corpus based on metadata (if applicable)
def filter_corpus(df, filters):
    """Apply filters to the corpus dataframe."""
    filtered_df = df.copy()
    
    # Convert the 'date' column to datetime
    if 'date' in filtered_df.columns:
        filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')

    # Apply date range filters
    if filters.get("start_date") and filters.get("end_date"):
        start_date = pd.to_datetime(filters['start_date'])
        end_date = pd.to_datetime(filters['end_date'])
        filtered_df = filtered_df[
            (filtered_df['date'] >= start_date) & 
            (filtered_df['date'] <= end_date)
        ]

    # Apply category filter
    if filters.get("category"):
        filtered_df = filtered_df[filtered_df['category'] == filters['category']]
    
    return filtered_df

# Main App
def main():
    st.set_page_config(page_title="Semantic Search Engine", layout="wide")

    st.title("🔍 Semantic Search Engine")

    # Sidebar settings
    st.sidebar.header("Settings")
    model_name = st.sidebar.selectbox(
        "Choose a SentenceTransformer model:",
        ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L12-v2", "distiluse-base-multilingual-cased-v1"]
    )
    top_k = st.sidebar.slider("Number of top results", 1, 20, 5)

    # Load model
    model = load_model(model_name)

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
        # Default corpus with metadata
        df = pd.DataFrame({
            "text": [
                "Machine learning is a subset of AI.",
                "Streamlit helps build data apps quickly.",
                "Transformers are powerful NLP models.",
                "Semantic similarity is useful for search.",
                "The sun rises in the east."
            ],
            "date": ["2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05"],
            "category": ["Technology", "Technology", "AI", "Search", "Science"]
        })
        st.info("Using default sample corpus with metadata.")

    # Validate corpus
    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column.")
        return

    # Optional filters
    st.sidebar.subheader("Filters")
    start_date = st.sidebar.date_input("Start Date", key="start_date")
    end_date = st.sidebar.date_input("End Date", key="end_date")
    category = st.sidebar.selectbox("Category", ["All"] + df['category'].unique().tolist())

    filters = {
        "start_date": start_date if start_date else None,
        "end_date": end_date if end_date else None,
        "category": category if category != "All" else None
    }

    # Apply filters
    df = filter_corpus(df, filters)

    # Preview corpus
    st.subheader("📄 Corpus Preview")
    st.dataframe(df)

    # Allow users to add text to the corpus
    st.sidebar.subheader("Add Text to Corpus")
    new_text = st.sidebar.text_area("Enter new text to add:")
    if st.sidebar.button("Add to Corpus") and new_text.strip():
        df = pd.concat([df, pd.DataFrame({"text": [new_text], "date": [pd.Timestamp.now()], "category": ["Uncategorized"]})], ignore_index=True)
        st.success("Text added to corpus!")

    # Compute corpus embeddings
    corpus = df["text"].tolist()
    corpus_embeddings = compute_corpus_embeddings(corpus, model)

    # Autocomplete suggestions
    st.sidebar.subheader("Autocomplete Suggestions")
    suggestions = [text[:50] for text in corpus]
    st.sidebar.write(suggestions)

    # Search query input
    query = st.text_input("Enter your search query:")
    if st.button("Search") and query:
        # Compute query embedding
        query_embedding = model.encode(query, convert_to_tensor=True)

        # Perform semantic search
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]

        # Display results
        st.subheader("🔎 Search Results")
        results = []
        for idx, hit in enumerate(hits):
            result_text = corpus[hit['corpus_id']]
            highlighted_text = highlight_query_words(result_text, query)
            score = hit['score']
            row = df.iloc[hit['corpus_id']]
            results.append({"Rank": idx + 1, "Text": result_text, "Score": score, "Date": row['date'], "Category": row['category']})
            st.markdown(f"**{idx + 1}.** {highlighted_text}", unsafe_allow_html=True)
            st.caption(f"Similarity Score: {score:.4f} | Date: {row['date']} | Category: {row['category']}")

        # Allow users to download results
        result_df = pd.DataFrame(results)
        csv = result_df.to_csv(index=False)
        st.download_button("Download Results as CSV", data=csv, file_name="search_results.csv", mime="text/csv")

# Run the app
if __name__ == "__main__":
    main()