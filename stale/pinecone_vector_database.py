import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from tqdm import tqdm
import pandas as pd
import os


class PineconeVectorDb:
    def __init__(self, config_filename , embedding_model , embeddings , api_key = None, env = None, index_name = None, embed_api_key = None):
        from config import Config
        #load the configuration file
        
        config = Config(os.path.join(os.getcwd() , config_filename))
        if embedding_model is not None or embedding_model not in config.get('embeddings', 'names'):
            print("EMBEDDING : Provide Embedding model name is not valid")
            return

        if api_key is None:
            api_key = os.environ.get("PINECONE_API_KEY")
            print('PINECONE: Loaded API key from environment variables.')
        if env is None:
            env = os.environ.get("PINECONE_ENV")
            print('PINECONE: Loaded environment from environment variables.')
        if index_name is None:
            index_name = os.environ.get("PINECONE_INDEX")
            print('PINECONE: Loaded index name from environment variables.')
        if embed_api_key is None:
            embed_api_key = os.environ.get(config.get('embeddings', embedding_model ))
            print(f"{embedding_model} : Loaded API key from environment variables.")

        pinecone.init(api_key=api_key, environment=env)
        print('PINECONE: initialized')
        
        self.index = pinecone.Index(index_name)
        print('PINECONE: Set index to - ', index_name)

        self.embeddings = embeddings
        print('Embedding model Loaded')

        self.vector_search = Pinecone(self.index, self.embeddings.embed_query, "text")
    
    def search(self, query, top_k=20):
        return self.vector_search.similarity_search(query, k=top_k)
    
    def upsert(self, data_path: str):
        """Upserts data into the vector database.
        Args:
            data_path (str): Path to the data file.
        """

        if '.csv' in data_path:
            data = pd.read_csv(data_path)
        elif '.parquet' in data_path:
            data = pd.read_parquet(data_path)        
        else:
            raise Exception('Data format not supported. Please provide a csv or parquet file.')
               
        for item in tqdm(data.values , desc = "Upserting" , unit = "row" , ncols = 100):
            # item - some ID , product id , merged product description  , metadata
            product_id = item[1]
            product_vector = self.embedding.embed_query(item[2])
            product_metadata = item[3]
            record_metadata = {"description" : item[2], "product_source" : str(product_metadata) }

            self.index.upsert(vectors =[{'product_id': product_id , 'values' : product_vector , 'metadata': record_metadata}])

        print("Product Data has been Upsered into Pinecone")
        