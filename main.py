from dataset import SRDataset
from metric import EdgeMetric

metric = EdgeMetric()
data = SRDataset("C:/SR/code/paper_metric_dataset/SR_dataset_subjective/dataset", \
    "./subjective_scores.json", banned_frames="./banned_frames.json", cases=["statue", "restaurant"])

print(len(data))
stats = data.__getitem__(50)

res = metric.forward(stats[0], stats[1], stats[2])

print(res, stats[3], stats[4])
