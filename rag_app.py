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
def split_text(text,chunk_size=256,chunk_overlap=20):
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