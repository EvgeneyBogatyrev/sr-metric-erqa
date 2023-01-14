import torch
from functools import partial
import cv2
import numpy as np
import albumentations as A


class ImageFormatter():

    edge_detector = partial(cv2.Canny, threshold1=100, threshold2=200)

    @staticmethod
    def format_image(image, edges=False):
        shape = image.shape
        if not edges:
            transform = A.Compose([A.Normalize()])
            transformed = transform(image=image)
            image = transformed["image"]
        image = torch.tensor(image.reshape(shape[2], shape[0], shape[1]))

        return image

    @staticmethod
    def match_edges(edge, gt_edge, shift_compensation=False, penalize_wider_edges=False):
        assert gt_edge.shape == edge.shape

        true_positive = np.zeros_like(edge)
        false_negative = gt_edge.copy()

        # Count true positive
        if shift_compensation:
            window_range = 1
        else:
            window_range = 0

        window = sorted(range(-window_range, window_range + 1), key=abs)  # Place zero at first place

        for i in window:
            for j in window:
                gt_ = np.roll(false_negative, i, axis=0)
                gt_ = np.roll(gt_, j, axis=1)

                ad = edge * gt_ * np.logical_not(true_positive)

                np.logical_or(true_positive, ad, out=true_positive)
                if penalize_wider_edges:
                    # Unmark already used edges
                    ad = np.roll(ad, -j, axis=1)
                    ad = np.roll(ad, -i, axis=0)
                    np.logical_and(false_negative, np.logical_not(ad), out=false_negative)

        if not penalize_wider_edges:
            false_negative = gt_edge * np.logical_not(true_positive)

        assert not np.logical_and(true_positive, false_negative).any()

        return true_positive, false_negative

    @staticmethod
    def get_edges_map(img1, img2):
        
        edges1 = ImageFormatter.edge_detector(img1) // 255
        edges2 = ImageFormatter.edge_detector(img2) // 255

        true_positive, false_negative = ImageFormatter.match_edges(edges1, edges2)

        blue = np.array([255, 0, 0])
        red = np.array([0, 0, 255])
        white = np.array([255, 255, 255])

        false_positive = edges2 - true_positive

        return false_negative[..., None] * blue[None, None] \
                + false_positive[..., None] * red[None, None] \
                + true_positive[..., None] * white[None, None]

    @staticmethod
    def format_input_paths(path1, path2):
        image1 = cv2.imread(path1)
        image2 = cv2.imread(path2)

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        edges_map = ImageFormatter.get_edges_map(image1, image2)

        image1 = ImageFormatter.format_image(image1)
        image2 = ImageFormatter.format_image(image2)
        edges_map = ImageFormatter.format_image(edges_map)

        return image1, image2, edges_map

    @staticmethod
    def format_input_images(image1, image2):
        edges_map = ImageFormatter.get_edges_map(image1, image2)

        image1 = ImageFormatter.format_image(image1)
        image2 = ImageFormatter.format_image(image2)
        edges_map = ImageFormatter.format_image(edges_map)

        return image1, image2, edges_map
