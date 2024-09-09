import os
import pickle
from pinecone import Pinecone
import voyageai
import pandas as pd
from tabulate import tabulate
# from not_used_embedding import EmbeddingModel
from embedder.hybridEmbeddingModel import HybridEmbeddingModel
from reranker.jinaAIMultiLingualReranker import JinaAIMultiLingualReranker
from dotenv import load_dotenv
from embedder.sparse.bmEmbedding import BmEmbedding

class RecommendationEngine:
    def __init__(self, pinecone_index_name, pinecone_apikey, sparse_embed_model, dense_embed_model, reranker_model):
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_apikey)                  
        self.index = pc.Index(pinecone_index_name)

        # Load sparse embedding model
        self.sparse_model = sparse_embed_model
        # Initialize dense embedding model
        self.dense_model = dense_embed_model
        # Initialize reranker
        self.reranker = reranker_model
        # Initialize embedding model
        self.hybrid_embedding_model = HybridEmbeddingModel(self.sparse_model, self.dense_model)


    def hybrid_scale(self, dense, sparse, alpha=0.5):
        """Hybrid vector scaling using a convex combination

            alpha * dense + (1 - alpha) * sparse

            Args:
                dense: Array of floats representing
                sparse: a dict of `indices` and `values`
                alpha: float between 0 and 1 where 0 == sparse only
                    and 1 == dense only
        """    
        from embedder.sparse.bmEmbedding import BmEmbedding
        

        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        hsparse = {
            'indices': sparse['indices'],
            'values': [v * (1 - alpha) for v in sparse['values']]
        }
        hdense = [v * alpha for v in dense]
        return hdense, hsparse

    def get_pretty_results(self, top_k_results):
        def truncate(string, length):
            return string if len(string) <= length else string[:length-3] + '...'
        
        flattened_data = []
        for product in top_k_results.matches:
            flattened_product = {'id': product['id'], **product['metadata']}
            flattened_data.append(flattened_product)
        
        df = pd.DataFrame(flattened_data)
        column_order = ['product_id', 'brand', 'title', 'color', 'locale']
        df = df[column_order]
         
        df['title'] = df['title'].apply(lambda x: truncate(x, 80))
        df['color'] = df['color'].apply(lambda x: truncate(x, 30))
        pretty_string = tabulate(df, headers='keys', tablefmt='grid', showindex=False)
        
        return df, pretty_string

    def batch_search(self, query_list, top_k=25, alpha=0.5):
        df_list = []
        pretty_string_list = []
        sparse_list, dense_list = self.hybrid_embedding_model.embed_queries(query_list)
        for i in range(len(sparse_list)):
            hdense , hsparse = self.hybrid_scale(dense_list[i] , sparse_list[i] , alpha)
            try : 
                if len(hdense) ==0 or len(hsparse) ==0 : 
                    print(f"Embedding model returned None values for Query : {query_list[i]}")
                    return -1
                top_k_results = self.index.query(
                    top_k=top_k,
                    vector=hdense,
                    sparse_vector=hsparse,
                    include_metadata=True
                )
                t_df,t_pretty_string = self.get_pretty_results(top_k_results ) 
                df_list.append(t_df)
                pretty_string_list.append(pretty_string_list)
            except:
                print(f"Embedding model returned None values for Query : {query_list[i]}")
                continue

        return df_list , pretty_string_list
        
    def search(self, query, top_k=25, alpha=0.5):
        sparse, dense = self.hybrid_embedding_model.embed_queries(query)
        if len(dense) == 1: dense = dense[0]
        if len(sparse) > 0  and len(dense)>0  :
            hdense, hsparse = self.hybrid_scale(dense, sparse, alpha)
        else: 
            print(f"Embedding model returned None values for Query : {query}")
            return -1
        
        top_k_results = self.index.query(
            top_k=top_k,
            vector=hdense,
            sparse_vector=hsparse,
            include_metadata=True
        )
        
        return self.get_pretty_results(top_k_results ) 

    def batch_rerank(self , query_list , search_results_list):
        rerank_scores_list = []
        for i in range(len(query_list)):
            rerank_scores_list.append(self.reranker.rerank_scores(query_list[i] , search_results_list[i]['title']))
        return rerank_scores_list

    def rerank_scores(self, query, docs):
        return self.reranker.rerank_scores(query, docs)
    
    def rerank_scores_in_batches(self , query_doc_pairlist , batch_size = 128):
        return self.reranker.rerank_scores_in_batches( query_doc_pairlist , batch_size = 128)

    def recommend(self, query, search_top_k=25, rerank_top_k = 10 ,  alpha=0.5 , batch = False):
        if batch:
            batch_search_results, batch_pretty_text = self.batch_search(query , search_top_k , alpha )
            batch_scores = self.batch_rerank(query , batch_search_results)
            for i in range(len(batch_search_results)):
                batch_search_results[i]['scores'] = batch_scores[i]
                batch_search_results[i] = batch_search_results[i].sort_values(by='scores' , ascending=False)
                batch_search_results[i] = batch_search_results[i][:rerank_top_k]
            return batch_search_results
        else:
            search_results , pretty_text = self.search(query, search_top_k, alpha)
            scores = self.rerank_scores(query, search_results['title'])
            search_results['scores'] = scores
            sorted_search_res = search_results.sort_values(by='scores' , ascending=False)
            return sorted_search_res[:rerank_top_k]
    
if __name__ == "__main__":
    # # Example usage
    # load_dotenv()
    # # pinecone VD index name
    # index_name = os.environ.get('PINECONE_INDEX_NAME')
    # print(f"Index name is {index_name}")
    # # pinecone -api -key 
    # pinecone_apikey = os.environ.get('PINECONE_API_KEY')
    # # setup sparse encoding model
    # sparse_embed_model = BmEmbedding()
    # sparse_embed_model.load_model(file_name = 'bm_fit_model.pkl')

    # # setup dense embed model
    # dense_embed_model = VoyageMultiLingualEmbedding(api_key = os.environ.get('VOYAGE_AI_Test'))
    
    # # reranker
    # reranker_model = JinaAIMultiLingualReranker()

    # engine = RecommendationEngine(index_name, pinecone_apikey, sparse_embed_model, dense_embed_model, reranker_model)
    
    # query = "dark blue french connection jeans for men"
    # recommendations = engine.recommend(query)
    
    # print("Query")
    # print(query)
    # print("Top recommendations:")
    # print(tabulate(recommendations, headers='keys', tablefmt='grid', showindex=False))

    # import os
    # import pickle
    # from pinecone import Pinecone
    # import pandas as pd
    # from tabulate import tabulate
    # # from not_used_embedding import EmbeddingModel
    # from embedder.hybridEmbeddingModel import HybridEmbeddingModel
    # from reranker.jinaAIMultiLingualReranker import JinaAIMultiLingualReranker
    # from dotenv import load_dotenv
    # from embedder.sparse.bmEmbedding import BmEmbedding
    # from embedder.dense.allminiLMEnEmbedding import AllMiniLMENEmbedding
    # from RecommendationEngine import RecommendationEngine

    # load_dotenv()
    # # pinecone VD index name
    # # index_name = os.environ.get('PINECONE_INDEX_NAME')
    # index_name = "product-search-en-allmini-langchain-pinecone"
    # print(f"Index name is {index_name}")
    # # pinecone -api -key 
    # pinecone_apikey = os.environ.get('PINECONE_GTE_API_KEY')
    # # setup sparse encoding model
    # sparse_embed_model = BmEmbedding()
    # sparse_embed_model.load_model(file_name = 'en_bm_fit_model.pkl')
    
    # # setup dense embed model
    # dense_embed_model = AllMiniLMENEmbedding()
    
    # # reranker
    # reranker_model = JinaAIMultiLingualReranker()

    # engine = RecommendationEngine(index_name, pinecone_apikey, sparse_embed_model, dense_embed_model, reranker_model)

    # query = "dark blue french connection jeans for men"
    # recommendations = engine.recommend(query, rerank_top_k= 10 , alpha= 0.8)

    # print("Query")
    # print(query)
    # print("Top recommendations:")
    # print(tabulate(recommendations, headers='keys', tablefmt='grid', showindex=False))

    import os
    import pickle
    from pinecone import Pinecone
    import pandas as pd
    from tabulate import tabulate
    # from not_used_embedding import EmbeddingModel
    from embedder.hybridEmbeddingModel import HybridEmbeddingModel
    from reranker.jinaAIMultiLingualReranker import JinaAIMultiLingualReranker
    from dotenv import load_dotenv
    from embedder.sparse.bmEmbedding import BmEmbedding
    from embedder.dense.allminiLMEnEmbedding import AllMiniLMENEmbedding
    from embedder.dense.voyageLargeEnInstructEmbedding import VoyageLargeEnInstrcutEmbedding
    
    load_dotenv()
    # pinecone VD index name
    # index_name = os.environ.get('PINECONE_INDEX_NAME')
    index_name = "product-search-en-voyage-langchain-pinecone"
    print(f"Index name is {index_name}")
    # pinecone -api -key 
    pinecone_apikey = os.environ.get('PINECONE_VOYAGE_EN_API_KEY')
    # setup sparse encoding model
    sparse_embed_model = BmEmbedding()
    sparse_embed_model.load_model(file_name = 'en_bm_fit_model.pkl')
    
    # setup dense embed model
    dense_embed_model = VoyageLargeEnInstrcutEmbedding(api_key = os.environ.get('VOYAGE_AI_Test'))
    
    # reranker
    reranker_model = JinaAIMultiLingualReranker()

    engine = RecommendationEngine(index_name, pinecone_apikey, sparse_embed_model, dense_embed_model, reranker_model)

    query = ["dark blue french connection jeans for men" , "dark blue french connection jeans for men", "dark blue french connection jeans for men" ]
    recommendations = engine.recommend(query, rerank_top_k= 10 , alpha= 0.8 , batch= True)
