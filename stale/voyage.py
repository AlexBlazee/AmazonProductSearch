from voyageai import Client
import numpy as np


class VoyageEmbedder():
    def __init__(self, api_key: str):
        self.client = Client(api_key)

    def embed(self, text: str | list[str]):
        if isinstance(text, str):
            return np.mean(self.client.embed(self.split_texts(text)).embeddings, axis=0).tolist()
        # else:
        #     return [np.mean(self.client.embed(self.split_texts(t)).embeddings, axis=0).tolist() for t in text]
        
    def split_texts(self, text: str, max_length: int = 4000):
        if self.count_tokens(text) <= max_length:
            return [text]
        return [text[:max_length]]
    
    def count_tokens(self, text: str):
        return self.client.count_tokens([text])