import numpy as np
from traitlets import Callable

from data_reader import DataReader
from multi_polynomial_regression import MultiPolynomialRegression
from tools import filter_lanes_coefs, line_plot, step_by_step_process_frame

def filter_points(points=np.zeros((0,3))):
  """Point filtering method.
    Behavior: Uses a threshold on intensity"""
  
  return points[points[:,3] >= 15]

def process_frame(data_reader: DataReader, model, frame_index: int = 0, filter_method: Callable = lambda x: x):
  """Processes a frame.
    Behavior: filter the points then fits a model to them and saves the closest lanes to the lane file."""
  
  if model is None:
    return
  
  model.points = filter_method(data_reader.read_points(frame_index))
  model.fit()

  lanes_coefs = filter_lanes_coefs(model.polynomial_family)

  data_reader.write_lanes_coefs(frame_index, lanes_coefs)

  return model.score

data_folder = "./pointclouds"
lanes_folder = "./sample_output"
num_point_attributes = 5

data_reader = DataReader(data_folder, lanes_folder, num_point_attributes)

model = MultiPolynomialRegression(
  min_delta=25,
  min_epsilon=1
)

scores = np.zeros((0,2))
for frame_index in range(data_reader.nb_frames):
  print(f'Processing frame {frame_index}')
  score = process_frame(data_reader, model, frame_index, filter_points)
  scores = np.append(scores, [[frame_index, score]], axis=0)

line_plot(scores, 'Frame index', 'Score', 'Score for each frame')

step_by_step_process_frame(data_reader, model, filter_points)

