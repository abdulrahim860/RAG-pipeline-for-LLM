import os 
import wikipedia
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def get_wikipedia_content(topic):
    try:
        page=wikipedia.page(topic)
        return page.content
    except wikipedia.exceptions.PageError:
        return None
    except wikipedia .exceptions.DisambiguationError as e:
        print(f"aAmbiguous topic. please be more specific. options:{e.options}")
        return None
topic=input('Enter a topic to learn about:')
document=get_wikipedia_content(topic)
if not document:
    print('could not retrieve information.')
    exit()

tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
def split_text(text,chunk_size=128,chunk_overlap=40):
    tokens=tokenizer.tokenize(text)
    chunks=[]
    start=0
    while start<len(tokens):
        end=min(start+chunk_size,len(tokens))
        chunks.append(tokenizer.convert_tokens_to_string(tokens[start:end]))
        if end==len(tokens):
            break
        start=end-chunk_overlap
    return chunks
chunks=split_text(document)
print(f"number of chunks:{len(chunks)}")

embedding_model=SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embeddings=embedding_model.encode(chunks)
dimension=embeddings.shape[1]
index=faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

query=input("Ask a question about the topic:")
query_embedding=embedding_model.encode([query])
k=3
distances,indices=index.search(np.array(query_embedding),k)
retrieved_chunks=[chunks[i] for i in indices[0]]
print("Retreived chunks:")
for chunk in retrieved_chunks:
    print("-"+chunk)

qa_model_name = 'deepset/roberta-base-squad2'
local_model_dir = './saved_model_roberta_squad2'

if not os.path.exists(local_model_dir):
    print("Model not found locally. Downloading and saving...")
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    qa_tokenizer.save_pretrained(local_model_dir)
    qa_model.save_pretrained(local_model_dir)
else:
    print("Loading model from local storage...")
    qa_tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(local_model_dir)

qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

context = "".join(retrieved_chunks)
answer = qa_pipeline(question=query, context=context)
print(f"Answer:{answer['answer']}")
