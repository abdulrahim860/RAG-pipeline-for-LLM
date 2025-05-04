import os
import wikipedia
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

@st.cache_data
def get_wikipedia_content(topic):
    try:
        page = wikipedia.page(topic)
        return page.content
    except wikipedia.exceptions.PageError:
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        st.warning(f"Ambiguous topic. Please be more specific. Options: {e.options}")
        return None

@st.cache_data
def split_text(text, chunk_size=128, chunk_overlap=40):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    tokens = tokenizer.tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokenizer.convert_tokens_to_string(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - chunk_overlap
    return chunks

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def create_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

@st.cache_resource
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

def main():
    st.title("🧠 Wikipedia RAG App")
    topic = st.text_input("Enter a topic to learn about:")

    if topic:
        document = get_wikipedia_content(topic)
        if not document:
            st.error("Could not retrieve information for this topic.")
            return

        st.success("Wikipedia content retrieved!")
        chunks = split_text(document)
        st.write(f"✅ Number of chunks created: {len(chunks)}")

        embedding_model = load_embedding_model()
        embeddings = embedding_model.encode(chunks)
        index = create_index(embeddings)

        query = st.text_input("Ask a question about this topic:")

        if query:
            query_embedding = embedding_model.encode([query])
            distances, indices = index.search(np.array(query_embedding), k=3)
            retrieved_chunks = [chunks[i] for i in indices[0]]
            st.subheader("📄 Retrieved Chunks:")
            for i, chunk in enumerate(retrieved_chunks, 1):
                st.text_area(f"Chunk {i}", chunk, height=100)

            qa_pipeline = load_qa_model()
            context = " ".join(retrieved_chunks)
            answer = qa_pipeline(question=query, context=context)
            st.subheader("💬 Answer:")
            st.write(answer['answer'])
            st.caption(f"Confidence Score: {answer['score']:.2f}")

if __name__ == "__main__":
    main()
