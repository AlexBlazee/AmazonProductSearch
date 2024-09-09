
class HybridEmbeddingModel:
    def __init__(self, sparse_model, dense_model):
        self.sparse_model = sparse_model
        self.dense_model = dense_model

    def embed_documents(self, docs):
        sparse = self.sparse_model.embed_documents(docs)
        dense = self.dense_model.embed_documents(docs)
        return sparse , dense

    def embed_queries(self , query):
        sparse = self.sparse_model.embed_query(query)
        dense = self.dense_model.embed_query(query)
        return sparse , dense
