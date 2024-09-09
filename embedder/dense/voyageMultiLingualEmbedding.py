import voyageai
from typing import List
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,  
)

class VoyageMultiLingualEmbedding:
    def __init__(self, api_key):
        self.api_key = api_key
        self.vo = voyageai.Client(api_key=api_key)

    # https://docs.voyageai.com/docs/embeddings
    # input_type = ["query" , "document" ]

    @retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(6))  
    def embed_with_backoff(self, **kwargs):
        return self.vo.embed(**kwargs)

    def embed_query(self,query):
        # dense = self.vo.embed(query, model="voyage-multilingual-2", input_type= "query", truncation=True).embeddings[0]
        # dense = self.embed_with_backoff(query, model="voyage-multilingual-2", input_type="query", truncation = True).embedings[0]
        if len(query) == 1: 
            dense = self.vo.embed(query, model="voyage-multilingual-2", input_type="query", truncation = True).embeddings[0]
        else : 
            dense = list(self.vo.embed(query, model="voyage-multilingual-2", input_type="query", truncation = True).embeddings)
        time.sleep(0.1)
        return dense
    
    def embed_documents(self, docs : List):
        dense = list(self.vo.embed(docs, model="voyage-multilingual-2", input_type="document", truncation=True).embeddings)
        # dense = list(self.embed_with_backoff(texts=docs, model="voyage-multilingual-2", input_type="document", truncation = True).embedings)
        # time.sleep(0.1)
        return dense