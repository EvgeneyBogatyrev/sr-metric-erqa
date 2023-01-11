from metric import EdgeMetric
import torch
from functools import partial
import cv2

def format_image(image):
    edge_detector = partial(cv2.Canny, threshold1=100, threshold2=200)
    edges = edge_detector(image)
    edges = torch.tensor(edges.reshape(1, 1, edges.shape[0], edges.shape[1]))
    shape = image.shape
    image = torch.tensor(image.reshape(1, shape[2], shape[0], shape[1]))

    return torch.cat((image, edges), dim=1)

dist = cv2.imread("dist.png")
dist2 = cv2.imread("dist.png")


dist = format_image(dist)
dist2 = format_image(dist2)

metric = EdgeMetric()

print(metric.forward(dist, dist2))