import multiprocessing
from functools import partial
from ranx import Qrels, Run, evaluate
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class Evaluator():

# Non parallel code and non batch code
	def evaluate_rank_metric( query_ids , data, engine,  TOP_VAL = None ):
		data = data.copy()
		data['esci_label']= data['esci_label'].replace({'E':2 , 'S':1})
		relevant_doc_ids_list = []
		relevant_scores_list = [] 
		recommended_doc_ids_list = []
		recommended_scores_list = []
		Iter_val = TOP_VAL if TOP_VAL  else len(query_ids)
		for i in range(Iter_val):
			temp = data[data['query_id'] == query_ids[i]].sort_values(by='esci_label' , ascending = False)
			relevant_doc_ids = temp['product_id'].to_list()
			relevant_doc_ids_list.append(relevant_doc_ids)
			relevant_score = temp['esci_label'].to_list()
			relevant_scores_list.append(relevant_score)
			query = temp['query'].iloc[0]
			recommendations = engine.recommend(query , rerank_top_k= 10 , alpha=0.75)
			recommended_doc_ids,recommended_scores = recommendations['product_id'].to_list(),recommendations['scores'].to_list()
			recommended_doc_ids_list.append(recommended_doc_ids)
			recommended_scores_list.append(recommended_scores)
		
		from ranx import Qrels, Run, evaluate
		qrels = Qrels()
		qrels.add_multi(
			q_ids= list(map(str, query_ids[:TOP_VAL])),
			doc_ids= relevant_doc_ids_list,
			scores= relevant_scores_list,
		)

		run = Run()
		run.add_multi(
			q_ids=list(map(str, query_ids[:TOP_VAL])),
			doc_ids=recommended_doc_ids_list,
			scores=recommended_scores_list,
		)
		eval_res = evaluate(qrels, run, ["hit_rate@1" , "hit_rate@5" , "hit_rate@10", "hits@1" , "hits@5" , "hits@10",   "precision@1" \
										 ,"precision@5" , "precision@10" ,"recall@1" , "recall@5" , "recall@10" ,"f1@1", "f1@5" , "f1@10", "mrr"])

		return relevant_doc_ids_list, relevant_scores_list , recommended_doc_ids_list , recommended_scores_list , eval_res

# relevant_doc_ids_list, relevant_scores_list , recommended_doc_ids_list , recommended_scores_list ,eval_res = evaluate(easy_queries_id , ES_df_queries_e , TOP_VAL= 5)