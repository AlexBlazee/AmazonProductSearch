from fastapi  import FastAPI as fsapi
from pydantic import BaseModel
from RecommendationEngine import RecommendationEngine
import os
from dotenv import load_dotenv
from embedder.sparse.bmEmbedding import BmEmbedding
from embedder.dense.voyageLargeEnInstructEmbedding import VoyageLargeEnInstrcutEmbedding
from reranker.jinaAIMultiLingualReranker import JinaAIMultiLingualReranker

app = fsapi()

# Load environment variables
load_dotenv()

# Initialize RecommendationEngine
index_name = "product-search-en-voyage-langchain-pinecone"
pinecone_apikey = os.environ.get('PINECONE_VOYAGE_EN_API_KEY')
sparse_embed_model = BmEmbedding()
sparse_embed_model.load_model(file_name='en_bm_fit_model.pkl')
dense_embed_model = VoyageLargeEnInstrcutEmbedding(api_key=os.environ.get('VOYAGE_AI_Test'))
reranker_model = JinaAIMultiLingualReranker()

engine = RecommendationEngine(index_name, pinecone_apikey, sparse_embed_model, dense_embed_model, reranker_model)

class SearchRequest(BaseModel):
    query: str
    rerank_top_k: int = 10
    alpha: float = 0.8

@app.post("/search")
def search_products(request: SearchRequest):
    recommendations = engine.recommend(request.query, rerank_top_k=request.rerank_top_k, alpha=request.alpha)
    return recommendations.to_dict(orient='records')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#  uvicorn api:app --reload