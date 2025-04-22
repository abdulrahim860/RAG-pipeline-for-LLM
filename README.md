# 🧠 RAG Pipeline for Query-Based Wikipedia Retrieval

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** pipeline using the **Wikipedia API** for knowledge retrieval. Given a user's query, the app fetches relevant content from Wikipedia, splits the text into manageable chunks, and prepares it for downstream usage like semantic search or question answering.

## 📌 Features

- 📖 Retrieves data from Wikipedia based on user input
- 🧱 Splits long text into overlapping token chunks
- 🔤 Uses `sentence-transformers/all-mpnet-base-v2` tokenizer
- ⚙️ Ready for integration with vector databases or LLMs
