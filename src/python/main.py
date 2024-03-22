import threading
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from traitlets import Callable

from data_reader import DataReader
from multi_polynomial_regression import MultiPolynomialRegression
from visualization import Visualization
from tools import filter_lanes_coefs, get_point_color_using_last_dimension, line_plot, process_all_frames, step_by_step_process_frame

def intensity_filter(points=np.zeros((0,3)), min_intensity=-np.inf, max_intensity=np.inf):
  """Point filtering using intensity"""
  
  filtered_points = points[points[:,3] >= min_intensity]
  filtered_points = filtered_points[filtered_points[:,3] <= max_intensity]

  return filtered_points

def height_filter(points=np.zeros((0,3)), min_height=-np.inf, max_height=np.inf):
  """Point filtering using height"""
  
  filtered_points = points[points[:,2] >= min_height]
  filtered_points = filtered_points[filtered_points[:,2] <= max_height]

  return filtered_points

def centered_height_filter(points=np.zeros((0,3)), min_height=-np.inf, max_height=np.inf):
  """Point filtering using height after centering around the median"""

  center_h = np.median(points[:,2])

  return height_filter(points, center_h + min_height, center_h + max_height)

data_folder = "./pointclouds"
lanes_folder = "./sample_output"
num_point_attributes = 5
data_reader = DataReader(data_folder, lanes_folder, num_point_attributes)

model = MultiPolynomialRegression(
  min_delta=25,
  min_epsilon=1
)

low_intesity_filter_threshold = 15
low_intesity_filter = lambda x: intensity_filter(x, low_intesity_filter_threshold)

def make_title(frame_index, pca_switch):
  title =  f'Visualizing frame {frame_index}'
  if pca_switch:
    title += ' after PCA'
  
  return title

def update(fig: plt.Figure, frame_index, pca_switch, data_reader,vis=Visualization()):
  title = make_title(frame_index, pca_switch)
  points = centered_height_filter(data_reader.read_points(frame_index), -1, 1)

  colored_points = np.c_[points[:,:3], np.clip(points[:,3] * 10, 0.0, 255.0)]
  colored_points = np.c_[points[:,:3], get_point_color_using_last_dimension(colored_points)]

  pca = PCA(n_components=3)
  pca_points = pca.fit_transform(points[:,:3])
  pca_points = np.c_[pca_points, colored_points[:,3:6]]
  # pca_points = pca.fit_transform(points[:,:5])
  # pca_points = np.c_[pca_points[:,:3], get_point_color_using_last_dimension(pca_points)]
  pca_lanes = pca.components_

  if fig is not None:
    plt.close(fig)
  # cumulative_variance = np.append([0], np.cumsum(pca.explained_variance_ratio_))
  # fig = line_plot(cumulative_variance, 'Nb of components', 'Explained variance ratio', 'Cumulative variance')
  intensities = points[:,2]
  intensities.sort()
  k=4.0
  x0=np.median(intensities)
  print(x0)
  scaled_intensities = 1.0 / (1.0 + np.exp(-k*(intensities - x0)))
  fig = line_plot(np.c_[intensities, scaled_intensities], 'Intensity', 'Scaled Intensity', '')
  # fig, ax = plt.subplots()

  # # We can set the number of bins with the *bins* keyword argument.
  # ax.hist(points[:,2] - np.median(points[:,2]), bins=200)
  # plt.show(block=False)

  
  display_points = np.zeros((0,6))
  if pca_switch:
    display_points = pca_points
  else:
    display_points = colored_points

  display_lanes = []
  if pca_switch:
    display_lanes = [[0.0,0.0], [1000.0,0.0]]
  else:
    display_lanes = pca_lanes[:2,:2]


  return vis.run_or_update(
    title = title,
    points = display_points,
    lanes_coefs = display_lanes
  ), fig

vis = Visualization()
fig: plt.Figure = None
frame_index = 0
pca_switch = False

while True:
  vis, fig = update(fig, frame_index, pca_switch, data_reader, vis)

  command = input().strip()
  if command == 'a':
    pca_switch = not pca_switch
  elif command == '+' and frame_index < data_reader.nb_frames - 1:
    frame_index += 1
  elif command == '-' and frame_index > 0:
    frame_index -= 1
  elif command == 'q':
    vis.stop()
    break

