import os
import numpy as np


class DataReader():
  """Class used to read and write the data from and to files"""

  def __init__(
    self,
    data_folder: str = "./pointclouds",
    lanes_folder: str = "./sample_output",
    num_point_attributes: int = 5
  ):
    """DataReader constructor:
    Inputs:
      data_foler: Folder containing the point clouds
      lanes_folder: Folder containing the lanes coefficients
      frame_index: Index of the frame to load"""
    
    self.data_folder: str = data_folder
    """Folder containing the lidar point clouds"""
    self.lanes_folder: str = lanes_folder
    """Folder containing the lanes coefficients"""
    self.num_point_attributes: int = num_point_attributes
    """Number of dimension of the points"""

    self.lidar_paths: list = None
    """Paths to the files containing the lidar points clouds"""
    self.lanes_paths: list = None
    """Paths to the files containing the lanes coefficients"""
    self.parse_folders()


  def parse_folders(self):
    """Parse data_folder and lanes_folder to get the  data to represent"""

    lidar_files = sorted(os.listdir(self.data_folder))
    self.lidar_paths = [os.path.join(self.data_folder, f) for f in lidar_files]
    self.nb_frames = len(self.lidar_paths)

    lanes_paths = []
    for lidar_file in lidar_files:
      lane_path = os.path.join(self.lanes_folder, lidar_file.replace("bin", "txt"))
      lanes_paths.append(lane_path)
    self.lanes_paths = lanes_paths


  def read_points(self, frame_index: int = 0):
    """Reads the points corresponding to the given frame index"""

    if self.lidar_paths is None or self.lidar_paths[frame_index] is None:
      return None
    
    return np.fromfile(self.lidar_paths[frame_index], dtype=np.float32).reshape(-1, self.num_point_attributes)

  def read_lanes_coefs(self, frame_index: int = 0):
    """Reads the lanes corresponding to the given frame index"""

    if self.lanes_paths is None or self.lanes_paths[frame_index] is None or not os.path.isfile(self.lanes_paths[frame_index]):
      print(f"Can't find lane file")
      return None
    
    lanes_coefs = []
    with open(self.lanes_paths[frame_index], "r") as f:
      for line in f:
        try:
          lane_coefs = [float(x) for x in line.strip().split(";")]
        except:
          lane_coefs = [0.0,0.0,0.0,0.0]
        lanes_coefs.append(lane_coefs)
    lanes_coefs = np.array(lanes_coefs)
    return lanes_coefs
  
  def write_lanes_coefs(self, frame_index: int = None, lanes_coefs=np.zeros((0,0))):
    """Writes the lanes to the file corresponding to the given frame index"""

    lanes_path = self.lanes_paths[frame_index]
    if lanes_path is not None:
      with open(lanes_path, "w") as f:
        out_str = '\n'.join([';'.join([f'{coef}' for coef in lane_coefs]) for lane_coefs in lanes_coefs])
        f.write(out_str)
        f.close()
    