import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import wikipedia
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

@st.cache_data
def get_wikipedia_content(topic):
    specific_pages = {
        "apple": "Apple Inc.",
        "samsung": "Samsung Electronics",
        "microsoft": "Microsoft"
    }
    topic_to_search = specific_pages.get(topic.lower(), topic)
    try:
        page = wikipedia.page(topic_to_search, auto_suggest=False)
        return page.content, topic_to_search
    except wikipedia.exceptions.DisambiguationError as e:
        for option in e.options:
            if "Inc" in option or "Electronics" in option:
                try:
                    page = wikipedia.page(option, auto_suggest=False)
                    return page.content, option
                except:
                    continue
        return None, topic_to_search
    except:
        return None, topic_to_search

def split_text(text, max_chunk_size=500, overlap=100):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence.split())
        if current_length + sentence_len > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]  
            current_length = sum(len(s.split()) for s in current_chunk)
        current_chunk.append(sentence)
        current_length += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

@st.cache_resource(show_spinner=False)
def load_qa_model(local_model_dir='./saved_model_roberta_squad2'):
    qa_model_name = 'deepset/roberta-base-squad2'
    if not os.path.exists(local_model_dir):
        qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
        qa_tokenizer.save_pretrained(local_model_dir)
        qa_model.save_pretrained(local_model_dir)
    else:
        qa_tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        qa_model = AutoModelForQuestionAnswering.from_pretrained(local_model_dir)
    return pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

def batch_encode(embedding_model, texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_emb = embedding_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(batch_emb)
    return np.vstack(embeddings)

def main():
    st.set_page_config(page_title="Improved Multi-Topic Wikipedia RAG", layout="wide")
    st.title("🧠 Improved Multi-Topic Wikipedia RAG App")

    topics_input = st.text_input("Enter multiple topics (comma-separated):")
    if not topics_input:
        st.info("Please enter one or more topics to fetch Wikipedia content.")
        return

    topics = [t.strip() for t in topics_input.split(",") if t.strip()]
    topic_contents = {}
    successful_topics = []
    failed_topics = []

    with st.spinner("Fetching Wikipedia articles..."):
        for topic in topics:
            content, true_name = get_wikipedia_content(topic)
            if content:
                topic_contents[true_name] = content
                successful_topics.append(true_name)
            else:
                failed_topics.append(topic)

    if not topic_contents:
        st.error("No valid Wikipedia articles found.")
        return

    st.success(f"✅ Articles loaded for: {', '.join(successful_topics)}")
    if failed_topics:
        st.warning(f"⚠️ Failed to fetch: {', '.join(failed_topics)}")

    embedding_model = load_embedding_model()
    topic_chunks = {}
    topic_embeddings = {}
    topic_indices = {}

    for topic, content in topic_contents.items():
        chunks = split_text(content)
        embeddings = batch_encode(embedding_model, chunks)
        index = create_faiss_index(embeddings)

        topic_chunks[topic] = chunks
        topic_embeddings[topic] = embeddings
        topic_indices[topic] = index

        page_url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        st.markdown(f"🔗 [{topic} Wikipedia Page]({page_url})")

    query = st.text_input("Ask a question across all topics:")

    if query:
        qa_pipeline = load_qa_model()
        query_embedding = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        for topic in successful_topics:
            st.markdown(f"### 🏷️ **{topic}**")
            index = topic_indices[topic]
            chunks = topic_chunks[topic]

            D, I = index.search(query_embedding, k=5)
            retrieved_chunks = [chunks[i] for i in I[0]]

            st.subheader("📄 Retrieved Chunks:")
            for i, chunk in enumerate(retrieved_chunks, 1):
                st.text_area(f"{topic} - Chunk {i}", chunk, height=120)

            context = " ".join(retrieved_chunks)
            result = qa_pipeline(question=query, context=context)

            if result['score'] > 0.2:
                st.subheader("💬 Answer:")
                st.write(result['answer'])
                st.caption(f"Confidence Score: {result['score']:.2f}")
            else:
                st.warning("Answer confidence too low; try rephrasing the question.")

if __name__ == "__main__":
    main()
