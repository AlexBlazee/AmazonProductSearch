from pinecone_text.sparse import BM25Encoder
from typing import List
import pickle
import os
# import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')

class BmEmbedding:
    def __init__(self) -> None:
        self.bm25 = BM25Encoder()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent_dir = os.path.dirname(self.current_dir)
        self.model_save_dir_name = "saved"
        self.model_save_dir_path = os.path.join(self.parent_dir, self.model_save_dir_name)

    def fit(self, corpus_data : List):
        self.bm25.fit(corpus_data)

    def save_model_as_pkl(self, file_name):
        if not os.path.exists(self.model_save_dir_path ):
            os.makedirs(self.model_save_dir_path )
        
        file_path = os.path.join(self.model_save_dir_path , file_name)

        with open(file_path, 'wb') as file:
            pickle.dump(self.bm25, file)

    def load_model(self , file_name):
        model_filepath = os.path.join(self.model_save_dir_path , file_name)
        with open(model_filepath, 'rb') as file:
            bm_fit_loaded = pickle.load(file)
        self.bm25 = bm_fit_loaded


    def embed_documents(self , docs ):
        sparse = self.bm25.encode_documents(docs)
        return sparse
    
    def embed_query(self , query):
        sparse = self.bm25.encode_queries(query)
        return sparse
