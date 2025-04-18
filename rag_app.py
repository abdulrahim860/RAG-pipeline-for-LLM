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
