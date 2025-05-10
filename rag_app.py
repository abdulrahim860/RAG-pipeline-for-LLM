import os
import wikipedia
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

@st.cache_data
def get_wikipedia_content(topic):
    specific_pages = {
        "apple": "Apple Inc.",
        "samsung": "Samsung Electronics",
        "microsoft": "Microsoft"
    }
    topic_to_search = specific_pages.get(topic.lower(), topic)
    try:
        page = wikipedia.page(topic_to_search)
        return page.content, topic_to_search
    except wikipedia.exceptions.PageError:
        return None, topic_to_search
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            first_option = e.options[0]
            page = wikipedia.page(first_option)
            return page.content, first_option
        except:
            st.warning(f"Ambiguous topic: {topic}. Options: {e.options}")
            return None, topic_to_search

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
    st.title("🧠 Multi-Topic Wikipedia RAG App")
    topics_input = st.text_input("Enter multiple topics (comma-separated):")

    if topics_input:
        topics = [t.strip() for t in topics_input.split(",") if t.strip()]
        topic_contents = {}
        successful_topics = []
        failed_topics = []
        topic_real_names = {}

        with st.spinner("Fetching Wikipedia articles..."):
            for topic in topics:
                content, searched_topic = get_wikipedia_content(topic)
                if content:
                    topic_contents[searched_topic] = content
                    successful_topics.append(searched_topic)
                    topic_real_names[searched_topic] = topic
                else:
                    failed_topics.append(topic)

        if not topic_contents:
            st.error("No valid Wikipedia articles found.")
            return

        st.success(f"✅ Articles loaded for: {', '.join(successful_topics)}")
        if failed_topics:
            st.warning(f"⚠️ Failed to fetch: {', '.join(failed_topics)}")

        topic_chunks = {}
        topic_embeddings = {}
        embedding_model = load_embedding_model()

        for topic, content in topic_contents.items():
            chunks = split_text(content)
            topic_chunks[topic] = chunks
            embeddings = embedding_model.encode(chunks)
            index = create_index(embeddings)
            topic_embeddings[topic] = (chunks, index)

            page_url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
            st.markdown(f"🔗 [{topic} Wikipedia Page]({page_url})")

        query = st.text_input("Ask a question across all topics:")

        if query:
            qa_pipeline = load_qa_model()

            for topic in successful_topics:
                st.markdown(f"### 🏷️ **{topic}**")

                sub_question = f"{query.strip().rstrip('?')} of {topic}?"

                chunks, index = topic_embeddings[topic]
                query_embedding = embedding_model.encode([sub_question])
                distances, indices = index.search(np.array(query_embedding), k=3)
                retrieved_chunks = [chunks[i] for i in indices[0]]

                st.subheader("📄 Retrieved Chunks:")
                for i, chunk in enumerate(retrieved_chunks, 1):
                    st.text_area(f"{topic} - Chunk {i}", chunk, height=100)

                context = " ".join(retrieved_chunks)
                answer = qa_pipeline(question=sub_question, context=context)

                st.subheader("💬 Answer:")
                st.write(answer['answer'])
                st.caption(f"Confidence Score: {answer['score']:.2f}")

if __name__ == "__main__":
    main()
