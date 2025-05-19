import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import wikipedia
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

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

@st.cache_data
def split_text(text, max_length=500):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_length:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence
    if chunk:
        chunks.append(chunk.strip())
    return chunks

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

@st.cache_resource
def create_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

@st.cache_resource
def load_qa_model(local_model_dir='./saved_model_bert_squad'):
    qa_model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
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
    st.title("🧠 Multi-Topic Wikipedia RAG App (Improved)")
    topics_input = st.text_input("Enter multiple topics (comma-separated):")

    if topics_input:
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

        for topic, content in topic_contents.items():
            chunks = split_text(content)
            topic_chunks[topic] = chunks
            embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
            index = create_index(embeddings)
            topic_embeddings[topic] = (chunks, index)

            page_url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
            st.markdown(f"🔗 [{topic} Wikipedia Page]({page_url})")

        query = st.text_input("Ask a question across all topics:")

        if query:
            qa_pipeline = load_qa_model()
            for topic in successful_topics:
                st.markdown(f"### 🏷️ **{topic}**")
                contextual_query = f"What is the answer to: {query.strip()} in the context of {topic}?"

                chunks, index = topic_embeddings[topic]
                query_embedding = embedding_model.encode([contextual_query], convert_to_numpy=True)
                distances, indices = index.search(query_embedding, k=5)

                if distances[0][0] > 1.5:
                    st.warning("Answer confidence too low; try rephrasing the question.")
                    continue

                retrieved_chunks = [chunks[i] for i in indices[0]]
                st.subheader("📄 Retrieved Chunks:")
                for i, chunk in enumerate(retrieved_chunks, 1):
                    st.text_area(f"{topic} - Chunk {i}", chunk, height=100)

                context = " ".join(retrieved_chunks)
                result = qa_pipeline(question=contextual_query, context=context)

                st.subheader("💬 Answer:")
                st.write(result['answer'])
                st.caption(f"Confidence Score: {result['score']:.2f}")

if __name__ == "__main__":
    main()
