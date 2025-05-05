# ML Dataset Semantic Search Engine

This is a **machine learning dataset search engine** built using **Streamlit, FAISS, and SentenceTransformers**. It allows users to search for ML datasets based on natural language queries, leveraging **NLP-based semantic similarity** for improved results.

## 🚀 Features
- **Semantic Search** using `SentenceTransformers` and `FAISS`
- **Intuitive UI** powered by Streamlit
- **Advanced Filtering** (task categories, languages, tags, licenses)
- **Cached Query Embeddings** for faster inference
- **Clickable Search Results** linking to Google

## 📦 Installation
Ensure you have Python installed (Python 3.7+ recommended). Then, run:

```bash
pip install streamlit pandas faiss-cpu sentence-transformers joblib
