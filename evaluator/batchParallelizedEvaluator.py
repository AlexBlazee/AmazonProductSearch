import multiprocessing
from functools import partial
from ranx import Qrels, Run, evaluate
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class BatchParallelizedEvaluator():
	def process_query(self, data, query_id_list , engine):
		relevant_doc_ids_l , relevant_score_l , query_l, recommended_doc_ids_l , recommended_scores_l = [],[],[],[],[]

		for query_id in query_id_list : 
			temp = data[data['query_id'] == query_id].sort_values(by='esci_label', ascending=False)
			relevant_doc_ids_l.append(list(temp['product_id'].values))
			relevant_score_l.append(list(temp['esci_label'].values))
			query_l.append(temp['query'].iloc[0])
		recommendations = engine.recommend(query_l, rerank_top_k=10, alpha=0.75 , batch= True)
		for i in range(len(recommendations)):
			recommended_doc_ids_l.append(recommendations[i]['product_id'].to_list())
			recommended_scores_l.append(recommendations[i]['scores'].to_list())
			time.sleep(0.3)
		return relevant_doc_ids_l, relevant_score_l, recommended_doc_ids_l, recommended_scores_l

	def evaluate_rank_metric(self ,query_ids, data, engine , TOP_VAL=None , batch_size = 128):
		Iter_val = TOP_VAL if isinstance(TOP_VAL, int) else len(query_ids)
		print("Queries under Search:", Iter_val)
		relevant_doc_ids_list = []
		relevant_scores_list = []
		recommended_doc_ids_list = []
		recommended_scores_list = []
		
		with ThreadPoolExecutor() as executor:
			futures = [executor.submit(self.process_query, data, query_ids[i:min(Iter_val , i + batch_size)] , engine ) for i in range(0 , Iter_val, batch_size)]
			for future in as_completed(futures):
				relevant_doc_ids, relevant_score, recommended_doc_ids, recommended_scores = future.result()
				relevant_doc_ids_list.extend(relevant_doc_ids)
				relevant_scores_list.extend(relevant_score)
				recommended_doc_ids_list.extend(recommended_doc_ids)
				recommended_scores_list.extend(recommended_scores)

		from ranx import Qrels, Run, evaluate
		qrels = Qrels()
		qrels.add_multi(
			q_ids=list(map(str, query_ids[:TOP_VAL])),
			doc_ids=relevant_doc_ids_list,
			scores=relevant_scores_list,
		)

		run = Run()
		run.add_multi(
			q_ids=list(map(str, query_ids[:TOP_VAL])),
			doc_ids=recommended_doc_ids_list,
			scores=recommended_scores_list,
		)

		eval_res = evaluate(qrels, run, ["hit_rate@1", "hit_rate@5", "hit_rate@10", "hits@1", "hits@5", "hits@10", 
										 "precision@1", "precision@5", "precision@10", "recall@1", "recall@5", 
										 "recall@10", "f1@1", "f1@5", "f1@10", "mrr"])

		return relevant_doc_ids_list, relevant_scores_list , recommended_doc_ids_list , recommended_scores_list ,eval_res 
		