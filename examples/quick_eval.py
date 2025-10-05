import mteb
from model2vec import StaticModel


def quick_eval(model_name="minishlab/potion-base-32M"):
    tasks = mteb.get_tasks(tasks=["CQADupstackEnglishRetrieval"])
    evaluator = mteb.MTEB(tasks)

    task = tasks[0]
    task.load_data()

    model = StaticModel.from_pretrained(model_name)

    results = evaluator.run(model, overwrite_results=True)
    scores = results[0].scores
    score_labels = [
        'ndcg_at_1',
        'ndcg_at_3',
        'ndcg_at_5',
        'ndcg_at_10',
        'ndcg_at_20',
        'ndcg_at_100',
        'ndcg_at_1000',
    ]

    for label in score_labels:
        print(f"{label}: {scores['test'][0][label]}")


if __name__ == "__main__":
    quick_eval()
