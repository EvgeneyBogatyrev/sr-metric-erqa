from dataset import SRDataset

data = SRDataset("C:/SR/code/paper_metric_dataset/SR_dataset_subjective/dataset", "./subjective_scores.json")

print(len(data))
stats = data.__getitem__(5)

print(stats[0].shape, stats[1].shape, stats[2].shape)
print(stats[3], stats[4])