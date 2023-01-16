from dataset import SRDataset, LIVEDataset
#from metric import EdgeMetric

#metric = EdgeMetric()
#data = SRDataset("C:/SR/code/paper_metric_dataset/SR_dataset_subjective/dataset", \
#    "./subjective_scores.json", banned_frames="./banned_frames.json", cases=["statue"])

#print(len(data))
#stats = data.__getitem__(1802)

#res = metric.forward(stats[0], stats[1], stats[2])

#print(res, stats[3], stats[4])

data = LIVEDataset("/main/mnt/calypso/25e_zim/metric/LIVE/images", "/main/mnt/calypso/25e_zim/metric/LIVE", cases=["bs"])

print(len(data))

chunk = data.__getitem__(3)

print(chunk[0][0], chunk[2][0])
print(chunk[1], chunk[3])
