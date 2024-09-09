from typing import List
from sentence_transformers import SentenceTransformer
import torch

class StellaEnEmbedding:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.query_prompt_name = "s2p_query"
        # search query, retrieve relevant passages that answer the query.\nQuery: {query}
        
        # self.query_prompt_name = "s2s_query"
        #Instruct: Retrieve semantically similar text.\nQuery: {query}
        
        self.model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True , device= self.device)

    def embed_query(self,query):
        dense = self.model.encode(query , convert_to_numpy= True , prompt_name=self.query_prompt_name)
        return dense
    
    def embed_documents(self, docs : List):
        dense =  self.model.encode(docs , convert_to_numpy= True)
        return dense