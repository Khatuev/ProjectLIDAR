# Standard library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

# Geospatial data processing
import laspy
import geopandas as gpd

# 3D and scientific computing
import open3d as o3d
from scipy.spatial import KDTree, cKDTree
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon, Point, box, LineString
import statsmodels.api as sm


# Image processing
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.color import label2rgb
from scipy.stats import binned_statistic_2d
from skimage.feature import peak_local_max
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.neighbors import NearestNeighbors

class LASProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = None
        self.points = None
        self.classifications = None
        self.ground_points = None
        self.non_ground_points = None
        self.normalized_points = None
        self.ground_elevations = None

    def load_data(self):
        try:
            self.raw_data = laspy.read(self.filepath)
            self.points = np.vstack((self.raw_data.x, self.raw_data.y, self.raw_data.z)).transpose()
            self.classifications = self.raw_data.classification
        except Exception as e:
            print(f"Failed to load data: {e}")

    def separate_points(self):
        self.ground_points = self.points[self.classifications == 2]
        self.non_ground_points = self.points[self.classifications != 2]

    def normalize_elevation(self):
        ground_df = pd.DataFrame(self.ground_points, columns=['x', 'y', 'z'])
        ground_df_mean = ground_df.groupby(['x', 'y']).mean().reset_index()
        F = griddata(ground_df_mean[['x', 'y']], ground_df_mean['z'], (self.points[:, 0], self.points[:, 1]), method='nearest')
        self.ground_elevations = F.flatten()
        self.normalized_points = np.copy(self.points)
        self.normalized_points[:, 2] -= self.ground_elevations

    def create_point_cloud(self, points, color=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if color is None:
            # Normalize height values to the 0-1 range for coloring
            z_values = points[:, 2] - points[:, 2].min()
            z_values /= z_values.max()
            cmap = plt.get_cmap('viridis')  # You can choose any colormap you like
            colors = cmap(z_values)[:, :3]  # Ignore alpha channel
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color(color)
        
        return pcd
    
    def display_color_map(self):
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)

        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmin=self.normalized_points[:, 2].min(), vmax=self.normalized_points[:, 2].max())
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
        cb.set_label('Elevation (normalized)')
        plt.show()

    def visualize(self, point_clouds):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()

        # Add point clouds to the visualizer
        for pcd in point_clouds:
            vis.add_geometry(pcd)
        
        # Define a callback for the ESC key to close the window
        def close_vis(vis):
            vis.destroy_window()
            return False  # Stops the visualization loop

        vis.register_key_callback(256, close_vis)  # 256 is the ASCII code for ESC
        vis.run()
        vis.destroy_window()

    def process(self):
        self.load_data()
        self.separate_points()
        self.normalize_elevation()
        
        ground_pcd = self.create_point_cloud(self.ground_points, [0, 0, 1])
        non_ground_pcd = self.create_point_cloud(self.non_ground_points, [1, 0, 0])
        
        # Visualize initial point clouds with ground and non-ground classification
        self.visualize([ground_pcd, non_ground_pcd])
        
        normalized_pcd = self.create_point_cloud(self.normalized_points)
        self.visualize([normalized_pcd])

        # Optionally, display the color map
        self.display_color_map()


class CanopyHeightModel:
    def __init__(self, las_processor, grid_size=200, min_distance=3):
        self.las_processor = las_processor
        self.grid_size = grid_size
        self.min_distance = min_distance
        self.bin_statistic = None
        self.coordinates = None
        self.x_edge = None
        self.y_edge = None

    def calculate_canopy_height(self):
        non_ground_normalized_points = self.las_processor.normalized_points[self.las_processor.classifications != 2]
        bin_statistic, x_edge, y_edge, binnumber = binned_statistic_2d(
            non_ground_normalized_points[:, 0],
            non_ground_normalized_points[:, 1],
            non_ground_normalized_points[:, 2],
            statistic='max',
            bins=[self.grid_size, self.grid_size]
        )
        self.bin_statistic = np.nan_to_num(bin_statistic)
        self.x_edge = x_edge
        self.y_edge = y_edge

    def detect_tree_tops(self):
        self.coordinates = peak_local_max(self.bin_statistic, min_distance=self.min_distance, labels=None)

    def get_tree_top_coordinates(self):
        x_bin_centers = (self.x_edge[:-1] + self.x_edge[1:]) / 2
        y_bin_centers = (self.y_edge[:-1] + self.y_edge[1:]) / 2
        x_coords = x_bin_centers[self.coordinates[:, 1]]
        y_coords = y_bin_centers[self.coordinates[:, 0]]
        return np.column_stack((x_coords, y_coords))

    def plot_canopy_height(self):
        plt.imshow(self.bin_statistic, origin='lower', cmap='viridis')
        plt.colorbar(label='Height (m)')
        plt.title('Canopy Height Model')
        plt.show()

    def plot_tree_tops(self):
        plt.imshow(self.bin_statistic, cmap='viridis', origin='lower')
        plt.plot(self.coordinates[:, 1], self.coordinates[:, 0], 'r.')
        plt.colorbar(label='Height (m)')
        plt.title('Canopy Height Model with Tree Tops')
        plt.show()

    def process(self):
        self.calculate_canopy_height()
        self.plot_canopy_height()
        self.detect_tree_tops()
        print(f"Number of detected tree tops: {len(self.coordinates)}")
        self.plot_tree_tops()
        actual_coords = self.get_tree_top_coordinates()

class TreeHeightCalculator:
    def __init__(self, las_processor, base_search_radius=1.0):
        self.las_processor = las_processor
        self.base_search_radius = base_search_radius
        if las_processor.normalized_points is not None:
            self.kd_tree = KDTree(las_processor.normalized_points[:, :2])
        else:
            raise ValueError("Normalized points not available. Ensure LASProcessor has run normalize_elevation().")

    def calculate_tree_heights(self, tree_coordinates):
        tree_heights = []
        for coord in tree_coordinates:
            local_density = len(self.kd_tree.query_ball_point(coord, self.base_search_radius))
            if local_density == 0:
                tree_heights.append(np.nan)
                continue
            adjusted_radius = self.base_search_radius * (1 + (10 / local_density))
            indices = self.kd_tree.query_ball_point(coord, adjusted_radius)
            if not indices:
                tree_heights.append(np.nan)
                continue
            heights = self.las_processor.normalized_points[indices, 2]
            combined_height = (np.max(heights) + np.median(heights) + np.mean(heights)) / 3
            tree_heights.append(combined_height)
        return tree_heights
    
class CrownRadiusCalculator:
    def __init__(self, points, pixel_size):
        self.points = points
        self.pixel_size = pixel_size
        self.tree = cKDTree(points[:, :2])

    def calculate_crown_radius(self, tree_top_coords, search_radius=5):
        indices = self.tree.query_ball_point(tree_top_coords, search_radius)
        canopy_area = len(indices) * self.pixel_size
        crown_radius = np.sqrt(canopy_area / np.pi)
        return crown_radius


class HeightEstimationMetrics:
    def __init__(self, reference_path, detected_gdf):
        self.reference_gdf = gpd.read_file(reference_path)
        self.detected_gdf = detected_gdf
        self.reference_points = np.array([(geom.x, geom.y) for geom in self.reference_gdf.geometry])
        self.reference_heights = self.reference_gdf['HRef'].to_numpy()
        self.detected_coords = np.array([(geom.x, geom.y) for geom in detected_gdf.geometry])
        self.detected_heights = detected_gdf['HRef'].to_numpy()

    def normalize_points(self):
        ref_min_x, ref_max_x = self.reference_points[:, 0].min(), self.reference_points[:, 0].max()
        ref_min_y, ref_max_y = self.reference_points[:, 1].min(), self.reference_points[:, 1].max()

        det_min_x, det_max_x = self.detected_coords[:, 0].min(), self.detected_coords[:, 0].max()
        det_min_y, det_max_y = self.detected_coords[:, 1].min(), self.detected_coords[:, 1].max()

        scale_x = (ref_max_x - ref_min_x) / (det_max_x - det_min_x)
        scale_y = (ref_max_y - ref_min_y) / (det_max_y - det_min_y)
        translation_x = ref_min_x - det_min_x * scale_x
        translation_y = ref_min_y - det_min_y * scale_y

        self.normalized_detected_coords = self.detected_coords * [scale_x, scale_y]
        self.translated_detected_coords = self.normalized_detected_coords + [translation_x, translation_y]
        self.detected_gdf['geometry'] = [Point(coord) for coord in self.translated_detected_coords]

    def sort_and_adjust(self):
        sorted_ref_indices = np.argsort(self.reference_heights)
        sorted_det_indices = np.argsort(self.detected_heights)

        self.sorted_reference_points = self.reference_points[sorted_ref_indices]
        self.sorted_reference_heights = self.reference_heights[sorted_ref_indices]

        self.sorted_detected_coords = self.translated_detected_coords[sorted_det_indices]
        self.sorted_detected_heights = self.detected_heights[sorted_det_indices]

        self.adjusted_coords = self.sorted_reference_points
        self.adjusted_heights = self.sorted_reference_heights

        adjusted_detected_data = {
            'DBHRef': np.zeros(len(self.adjusted_coords)),
            'HRef': self.sorted_detected_heights,
            'VolRef': np.zeros(len(self.adjusted_coords)),
            'geometry': [Point(coord) for coord in self.adjusted_coords]
        }

        self.adjusted_detected_gdf = gpd.GeoDataFrame(adjusted_detected_data, geometry='geometry')

    def calculate_metrics(self):
        self.mae = np.mean(np.abs(self.adjusted_heights - self.sorted_detected_heights))
        self.rmse = np.sqrt(mean_squared_error(self.adjusted_heights, self.sorted_detected_heights))
        self.mape = np.mean(np.abs((self.adjusted_heights - self.sorted_detected_heights) / self.adjusted_heights)) * 100
        self.r2 = r2_score(self.adjusted_heights, self.sorted_detected_heights)
        self.bias = np.mean(self.sorted_detected_heights - self.adjusted_heights)
        self.si = self.rmse / np.mean(self.adjusted_heights)
        self.pearson_corr = np.corrcoef(self.adjusted_heights, self.sorted_detected_heights)[0, 1]
        self.errors = self.sorted_detected_heights - self.adjusted_heights

    def filter_outliers(self):
        Q1 = np.percentile(self.errors, 25)
        Q3 = np.percentile(self.errors, 75)
        IQR = Q3 - Q1
        outlier_threshold = 1.5 * IQR

        filtered_indices = np.where((self.errors >= Q1 - outlier_threshold) & (self.errors <= Q3 + outlier_threshold))[0]
        self.filtered_matched_points = pd.DataFrame({
            'HRef_ref': self.adjusted_heights[filtered_indices],
            'HRef_detected': self.sorted_detected_heights[filtered_indices]
        })
        self.filtered_errors = self.errors[filtered_indices]

    def regression_analysis(self):
        X_filtered = sm.add_constant(self.filtered_matched_points['HRef_ref'])
        self.model_filtered = sm.OLS(self.filtered_matched_points['HRef_detected'], X_filtered).fit()
        print(self.model_filtered.summary())

        plt.figure(figsize=(10, 6))
        plt.scatter(self.filtered_matched_points['HRef_ref'], self.filtered_matched_points['HRef_detected'], alpha=0.5)
        plt.plot(self.filtered_matched_points['HRef_ref'], self.model_filtered.predict(X_filtered), 'r--', label='Regression Line')
        plt.xlabel('Reference Heights (m)')
        plt.ylabel('Detected Heights (m)')
        plt.title('Regression Analysis of Detected vs Reference Heights (Without Outliers)')
        plt.legend()
        plt.show()

    def plot_error_distribution(self):
        plt.figure(figsize=(10, 6))
        plt.hist(self.filtered_errors, bins=20, color='blue', edgecolor='black')
        plt.title('Distribution of Errors in Tree Height Estimation (Without Outliers)')
        plt.xlabel('Error (m)')
        plt.ylabel('Frequency')
        plt.show()

        height_ranges = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]
        range_labels = ['0-5m', '5-10m', '10-15m', '15-20m', '20-25m', '25-30m']
        filtered_errors_in_ranges = {label: [] for label in range_labels}

        for ref_height, error in zip(self.filtered_matched_points['HRef_ref'], self.filtered_errors):
            for i, (low, high) in enumerate(height_ranges):
                if low <= ref_height < high:
                    filtered_errors_in_ranges[range_labels[i]].append(error)

        plt.figure(figsize=(12, 8))
        plt.boxplot([filtered_errors_in_ranges[label] for label in range_labels], labels=range_labels)
        plt.title('Error Distribution in Different Height Ranges (Without Outliers)')
        plt.xlabel('Height Range (m)')
        plt.ylabel('Error (m)')
        plt.show()

    def plot_movement_lines(self):
        lines = [LineString([self.sorted_detected_coords[i], self.adjusted_coords[i]]) for i in range(len(self.adjusted_coords))]
        lines_gdf = gpd.GeoDataFrame(geometry=lines)

        fig, ax = plt.subplots(figsize=(12, 10))
        self.reference_gdf.plot(ax=ax, color='blue', marker='o', label='Reference Points')
        bounding_box_gdf.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2, label='Bounding Box') # type: ignore
        grid_gdf.boundary.plot(ax=ax, color='grey', linewidth=0.5, linestyle='--', label='Grid') # type: ignore
        self.detected_gdf.plot(ax=ax, color='green', marker='x', label='Original Detected Points')
        self.adjusted_detected_gdf.plot(ax=ax, color='purple', marker='x', label='Adjusted Detected Points')
        lines_gdf.plot(ax=ax, color='black', linewidth=0.1, linestyle='-', label='Movement Lines')

        plt.title("Movement of Detected Points to Match Reference Points")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.show()

    def plot_cdf_errors(self):
        sorted_errors = np.sort(self.filtered_errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

        plt.figure(figsize=(10, 6))
        plt.plot(sorted_errors, cdf, marker='.', linestyle='none')
        plt.xlabel('Error (m)')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Function (CDF) of Errors')
        plt.grid(True)
        plt.show()

    def run_pipeline(self):
        self.normalize_points()
        self.sort_and_adjust()
        self.calculate_metrics()
        self.filter_outliers()
        self.regression_analysis()
        self.plot_error_distribution()
        self.plot_movement_lines()
        self.plot_cdf_errors()