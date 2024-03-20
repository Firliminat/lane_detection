from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from data_reader import DataReader
from multi_polynomial_regression import MultiPolynomialRegression
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

print('Enter the frame index then press ENTER to procede.')
frame_index = int(input().strip())
print(frame_index)

points = data_reader.read_points(frame_index)
lanes_coefs = data_reader.read_lanes_coefs(frame_index)

vis = Visualization(f'Visualization of frame {frame_index}', points, lanes_coefs)
vis.run()

print('Next step: Data filtering. Press ENTER to procede.')
input()

def filter_points(points=np.zeros((0,3))):
  return points[points[:,3] >= 15]

points = filter_points(points)
vis.update(
  'Points filtering of frame {frame_index}',
  np.c_[points[:,:3], get_point_color_using_last_dimension(points)],
  lanes_coefs
)

print('Next step: Fitting the model. Press ENTER to procede.')
input()

def loop_callback_wrapper(points=np.zeros((0,3))):
  def loop_callback(model: MultiPolynomialRegression):
    points[:,4] = model.points[:,2]
    lanes_coefs = model.polynomial_family
    vis.update(
      'MultiPolynomial Regression fitting to frame {frame_index}',
      np.c_[points[:,:3], get_point_color_using_last_dimension(points)],
      lanes_coefs
    )
  return loop_callback

model = MultiPolynomialRegression(deg=3, points=points[:,0:2])
model.fit(polynomials_fitting_callback=loop_callback_wrapper(points), verbose=True)

input()