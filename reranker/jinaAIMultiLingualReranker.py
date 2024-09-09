from sentence_transformers import CrossEncoder
import torch

class JinaAIMultiLingualReranker:
    def __init__(self ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CrossEncoder( "jinaai/jina-reranker-v2-base-multilingual",
                        automodel_args={"torch_dtype": "auto"},
                        trust_remote_code=True,
                        device = device
                    )
        
    def rerank_scores(self, query, docs):
        """
        query : The product search query
        docs  : List of documents/products as type list
        """
        sentence_pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(sentence_pairs, convert_to_tensor=True).tolist()
        return scores

    def rerank_scores_in_batches(self , query_doc_pairlist , batch_size = 128):
        """
        query_doc_pairlist should be of the form: 
             [
                (query1 , document1 ),
                (query2 , document2 ),
                 ....
             ]
        """
        scores = []
        for i in range(0, len(query_doc_pairlist), batch_size):
            batch = query_doc_pairlist[i:i+batch_size]
            batch_scores = self.model.predict(batch)
            scores.extend(batch_scores)
        return scores
