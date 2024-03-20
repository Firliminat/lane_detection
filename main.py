
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from visualization import Visualization

def read_data(data_folder, lane_folder):
  lidar_files = sorted(os.listdir(data_folder))
  lidar_paths = [os.path.join(data_folder, f) for f in lidar_files]
  lane_paths = []
  for lidar_file in lidar_files:
    lane_path = os.path.join(lane_folder, lidar_file.replace("bin", "txt"))
    if os.path.isfile(lane_path):
      lane_paths.append(lane_path)
    else:
      lane_paths.append(None)
  return lidar_paths, lane_paths

def read_points(lidar_path, num_point_attributes):
  return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, num_point_attributes)

def read_lanes_coefs(lanes_path):
  if lanes_path is not None:
    lanes_coef = []
    with open(lanes_path, "r") as f:
      for line in f:
        try:
          lane_coefs = [float(x) for x in line.strip().split(";")]
        except:
          lane_coefs = [0.0,0.0,0.0,0.0]
        lanes_coef.append(lane_coefs)
      lanes_coef = np.array(lanes_coef)
    return lanes_coef
  
  print(f"Can't find lane file")
  return None

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
frame_index = 0
num_point_attributes = 5

lidar_paths, lanes_paths = read_data(data_folder, lanes_folder)

points = read_points(lidar_paths[frame_index], num_point_attributes)
lanes_coefs = read_lanes_coefs(lanes_paths[frame_index])

print('1st display')
vis = Visualization('Test', points, lanes_coefs)
vis.run()
input()
print('Update once')
vis.update(
  'Updated once',
  np.c_[points[:,:3], get_point_color_using_last_dimension(points[:,:4])],
  np.array([[0,0,1,0],[0,0,0,1]])
)
input()
print('Update twice')
vis.update(
  'Updated twice',
  np.c_[points[:,:3], get_point_color_using_last_dimension(points)],
  np.array([[0,0,-1,0],[0,0,0,-1]])
)
input()
print('Stop')
vis.stop()
print('Stoped')