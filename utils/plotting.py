import numpy as np
import cv2
from matplotlib import pyplot as plt

def AddHeatmap(img, heatmap):
    '''
    :param img:  np.array  (H,W,3) RGB
    :param heatmap: np.array  (H,W)  OK to be not normalized
    :return:
    '''
    # normalize heatmap

    heatmap = np.maximum(heatmap, 0)
    heatmap = 1.*(heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize between 0-1
    heatmap = np.uint8(heatmap * 255)  # Scale between 0-255 to visualize

    heatmap = np.repeat(heatmap[:, :, np.newaxis], 3, axis=2)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HSV)

    img_with_heatmap = np.float32(heatmap) + np.float32(img)

    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    img_with_heatmap = np.uint8(255 * img_with_heatmap)
    return img_with_heatmap