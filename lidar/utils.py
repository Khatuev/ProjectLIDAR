import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.neighbors import KDTree, NearestNeighbors
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

def compute_accuracy(detected_tree_tops, reference_points, threshold_distance=5.0):
    ref_tree = KDTree(reference_points)
    distances, _ = ref_tree.query(detected_tree_tops)
    accuracy = np.sum(distances <= threshold_distance) / len(detected_tree_tops)
    return accuracy

def calculate_mae(predicted, actual):
    return np.mean(np.abs(predicted - actual))

def normalize_points(reference_points, detected_points):
    ref_min_x, ref_max_x = reference_points[:, 0].min(), reference_points[:, 0].max()
    ref_min_y, ref_max_y = reference_points[:, 1].min(), reference_points[:, 1].max()
    det_min_x, det_max_x = detected_points[:, 0].min(), detected_points[:, 0].max()
    det_min_y, det_max_y = detected_points[:, 1].min(), detected_points[:, 1].max()

    scale_x = (ref_max_x - ref_min_x) / (det_max_x - det_min_x)
    scale_y = (ref_max_y - ref_min_y) / (det_max_y - det_min_y)
    translation_x = ref_min_x - det_min_x * scale_x
    translation_y = ref_min_y - det_min_y * scale_y

    scaled_detected_points = detected_points * [scale_x, scale_y]
    translated_detected_points = scaled_detected_points + [translation_x, translation_y]
    normalized_reference_points = reference_points * [scale_x, scale_y] + [translation_x, translation_y]

    return translated_detected_points, normalized_reference_points

def plot_distribution(data, title, xlabel, ylabel, bins=20, color='blue'):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_regression(x, y, model, xlabel, ylabel, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5)
    plt.plot(x, model.predict(sm.add_constant(x)), 'r--', label='Regression Line')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def filter_outliers(errors, threshold=10):
    Q1 = np.percentile(errors, 25)
    Q3 = np.percentile(errors, 75)
    IQR = Q3 - Q1
    outlier_threshold = 1.5 * IQR

    filtered_indices = np.where((errors >= Q1 - outlier_threshold) & (errors <= Q3 + outlier_threshold))[0]
    return filtered_indices