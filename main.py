
from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from data_reader import DataReader
from visualization import Visualization

def get_point_color_using_last_dimension(points):
  scaler = preprocessing.MinMaxScaler(feature_range=(0,255)).fit(points)
  color_values = scaler.transform(points)
  color_values = np.clip(color_values[:, -1], 0, 255)
  color_values = color_values.astype(np.uint8)
  cmap = plt.get_cmap("viridis")

  # Initialize the matplotlib color map
  sm = plt.cm.ScalarMappable(cmap=cmap)

  # Obtain linear color range
  color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

  color_range = color_range.reshape(256, 3).astype(np.float32) / 255.0
  colors = color_range[color_values]
  return colors

data_folder = "./pointclouds"
lanes_folder = "./sample_output"
num_point_attributes = 5

data_reader = DataReader(data_folder, lanes_folder, num_point_attributes)

frame_index = 0

points = data_reader.read_points(frame_index)
lanes_coefs = data_reader.read_lanes_coefs(frame_index)

print('1st display')
vis = Visualization('Test', points, lanes_coefs)
vis.run()

input()
print('Update once')

frame_index = 1
points = data_reader.read_points(frame_index)
lanes_coefs = np.array([[0,0,1,0],[0,0,0,1]])

vis.update(
  'Updated once',
  np.c_[points[:,:3], get_point_color_using_last_dimension(points)],
  lanes_coefs
)

input()
print('Update twice')

frame_index = 2
points = data_reader.read_points(frame_index)
lanes_coefs = np.array([[0,0,-1,0],[0,0,0,-1]])

vis.update(
  'Updated twice',
  np.c_[points[:,:3], get_point_color_using_last_dimension(points)],
  lanes_coefs
)

input()
print('Stop')

vis.stop()

print('Stoped')