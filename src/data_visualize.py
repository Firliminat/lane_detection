import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import threading
from pynput import keyboard



class Vis():
  def __init__(self, data_folder, lane_folder):
    
    # Points filtering variables
    self.lidar_beam_value = -1.0
    self.intensity_threshold = 15

    self.index = 0
    self.num_point_attributes = 5
    self.lidar_paths, self.lane_paths = self.read_data(data_folder, lane_folder)
    self.frame_length = len(self.lidar_paths)
    self.points = self.load_points()
    self.lanes = self.load_lanes()

  def filter_points(self, raw_points):
    """Filter raw_points based on their intensity and lidar beam value"""
    filtered_points = raw_points
    if self.lidar_beam_value > -1.0:
      filtered_points = filtered_points[filtered_points[:,4] == self.lidar_beam_value]
    if self.intensity_threshold > 0:
      filtered_points = filtered_points[filtered_points[:,3] >= self.intensity_threshold]
    return filtered_points
  
  def load_points(self):
    raw_points = np.fromfile(self.lidar_paths[self.index], dtype=np.float32).reshape(-1, self.num_point_attributes)

    filtered_points = self.filter_points(raw_points)
    xyz = filtered_points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    colors = self.get_point_color_using_intensity(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
  
  def load_lanes(self):
    lane_path = self.lane_paths[self.index]
    if lane_path is not None:
      with open(lane_path, "r") as f:
        left_lane_coef = f.readline()
        left_lane_coef = [float(x) for x in left_lane_coef.strip().split(";")]
        right_lane_coef = f.readline()
        right_lane_coef = [float(x) for x in right_lane_coef.strip().split(";")]
      connect = []
      x_max = 40
      num = 80
      xs = np.linspace(start=-x_max, stop=x_max, num=num, endpoint=True)
      left_ys = left_lane_coef[-1]
      power = len(left_lane_coef) - 1
      for i, coef in enumerate(left_lane_coef[:-1]):
        left_ys += np.power(xs, power - i) * coef

      right_ys = right_lane_coef[-1]
      for i, coef in enumerate(right_lane_coef[:-1]):
        right_ys += np.power(xs, power - i) * coef

      left_lane = np.stack([xs, left_ys, np.zeros_like(xs, dtype=np.float32)], axis=-1)
      right_lane = np.stack([xs, right_ys, np.zeros_like(xs, dtype=np.float32)], axis=-1)
      lane = np.concatenate([left_lane, right_lane], axis=0)

      connect = [[i, i + 1] for i in range(len(xs) - 1)] +\
                [[i, i + 1] for i in range(len(xs), 2 * len(xs) - 1)]
      connect = np.array(connect)

      lines = o3d.geometry.LineSet()
      lines.points = o3d.utility.Vector3dVector(lane)
      lines.lines = o3d.utility.Vector2iVector(connect)
      return lines
    print(f"Can't find lane file")
    return None

  def read_data(self, data_folder, lane_folder):
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

  def read_points(self, lidar_path):
    return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)

  def get_point_color_using_intensity(self, points):
    scale_factor = 10
    scaled_intensity = np.clip(points[:, 3] * scale_factor, 0, 255)
    scaled_intensity = scaled_intensity.astype(np.uint8)
    cmap = plt.get_cmap("viridis")

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    color_range = color_range.reshape(256, 3).astype(np.float32) / 255.0
    colors = color_range[scaled_intensity]
    return colors

  def visualize(self):
    gui.Application.instance.initialize()
    self.window = gui.Application.instance.create_window(self.make_title(), 1280, 720)
    self._3d = gui.SceneWidget()
    self._3d.scene = rendering.Open3DScene(self.window.renderer)
    self._3d.scene.set_background([0.3, 0.3, 0.3, 1.0])
    self.window.add_child(self._3d)

    bounds = self.points.get_axis_aligned_bounding_box()
    self._3d.setup_camera(60, bounds, bounds.get_center())
    self.update_points()
    self.update_lanes()

    self.update_geometry_from_input()
    gui.Application.instance.run()

  def update_geometry_from_input(self):
    def update_geometry_thread():
        def on_press(key):
            if key == keyboard.Key.left and self.index > 0:
                self.index -= 1
            elif key == keyboard.Key.right and self.index < self.frame_length - 1:
                self.index += 1
            elif key == keyboard.Key.down and self.lidar_beam_value > -1.0:
                self.lidar_beam_value -= 1.0
            elif key == keyboard.Key.up and self.lidar_beam_value < 63.0:
                self.lidar_beam_value += 1.0
            elif key == keyboard.Key.page_down and self.intensity_threshold > 0:
                self.intensity_threshold -= 1
            elif key == keyboard.Key.page_up and self.intensity_threshold < 255:
                self.intensity_threshold += 1
            else:
              return
            
            self.points = self.load_points()
            self.lanes = self.load_lanes()
            gui.Application.instance.post_to_main_thread(
              self.window,
              self.update
            )

        with keyboard.Listener(
            on_press=on_press) as listener:
            listener.join()

    threading.Thread(target=update_geometry_thread, daemon=True).start()

  def update_points(self):
    name = "__points__"
    if self._3d.scene.scene.has_geometry(name):
        self._3d.scene.remove_geometry(name)
    self._3d.scene.add_geometry(name, self.points, rendering.MaterialRecord())

  def update_lanes(self):
    name = "__lanes__"
    if self._3d.scene.scene.has_geometry(name):
      self._3d.scene.remove_geometry(name)
    if self.lanes is not None:
      mat = rendering.MaterialRecord()
      mat.shader = "unlitLine"
      mat.line_width = 2 * self.window.scaling
      self._3d.scene.add_geometry(name, self.lanes, mat)

  def make_title(self):
    return f"Frame Index: {self.index} / {self.frame_length - 1} - {self.lidar_paths[self.index]} | Lidar Beam Value : {self.lidar_beam_value} | Intensity Threshold : {self.intensity_threshold}"

  def update_title(self):
    self.window.title = self.make_title()
  
  def update(self):
    self.update_title()
    self.update_points()
    self.update_lanes()
    self.window.post_redraw()


if __name__ == "__main__":
  data_folder = "./pointclouds"
  lane_folder = "./sample_output"
  vis = Vis(data_folder, lane_folder)
  vis.visualize()
