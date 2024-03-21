
from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from traitlets import Callable

from data_reader import DataReader
from multi_polynomial_regression import MultiPolynomialRegression
from visualization import Visualization

def get_point_color_using_last_dimension(points):
  """Returns colors based on last dimensions of the array"""

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

def line_plot(array, xlabel, ylabel, title):
  """Creates a line plot with array[:,0] as X axis and array[:,1] as Y axis"""

  if len(array.shape) < 2 or array.shape[1] < 2:
    array = np.c_[[[i] for i in range(array.shape[0])], array]
  
  fig, ax = plt.subplots()
  ax.plot(array[:,0], array[:,1])

  ax.set(
    xlabel=xlabel,
    ylabel=ylabel,
    title=title
  )
  ax.grid()

  plt.show(block=False)
  
  return fig

def filter_lanes_coefs(lanes_coefs=np.zeros((0,4))):
  """Lanes filtering method.
    Behavior: Keeps the 2 lanes closest to the origin"""
  
  distances_to_origin = np.array([MultiPolynomialRegression.dist_to_polynomial(np.zeros(2),lane_coefs) for lane_coefs in lanes_coefs])
  sorted_lanes_coefs = lanes_coefs[distances_to_origin.argsort()]

  return sorted_lanes_coefs[:2]

def step_by_step_process_frame(data_reader: DataReader, model, filter_method: Callable = lambda x: x):
  print('Enter the frame index then press ENTER to procede.')
  frame_index = int(input().strip())

  points = data_reader.read_points(frame_index)
  lanes_coefs = data_reader.read_lanes_coefs(frame_index)

  vis = Visualization(f'Visualization of frame {frame_index}', points, lanes_coefs)
  vis.run()

  print('Next step: Data filtering. Press ENTER to procede.')
  input()

  points = filter_method(points)
  vis.update(
    'Points filtering of frame {frame_index}',
    np.c_[points[:,:3], get_point_color_using_last_dimension(points)],
    lanes_coefs
  )

  print('Next step: Fitting the model. Press ENTER to procede.')
  input()

  model.points = points

  def loop_callback_wrapper(points=np.zeros((0,3))):
    def loop_callback(model: MultiPolynomialRegression):
      points[:,4] = model.points[:,2]
      new_lanes_coefs = model.polynomial_family
      vis.update(
        'MultiPolynomial Regression fitting to frame {frame_index}',
        np.c_[points[:,:3], get_point_color_using_last_dimension(points)],
        new_lanes_coefs
      )
    return loop_callback

  model.fit(
    score_plotting=True,
    polynomials_fitting_callback=loop_callback_wrapper(model.points),
    verbose=True
  )

  input()