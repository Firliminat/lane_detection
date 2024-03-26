import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import threading
from pynput import keyboard
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



class Vis():
  def __init__(self, data_folder, lane_folder):
    
    # Points filtering variables
    self.lidar_beam_value = -1.0
    self.intensity_threshold = 0
    self.intensity_ceiling = 255
    self.x_ceiling = 40
    self.y_ceiling = 8

    # clustering variables
    self.max_clusters = 1
    self.cluster_dim = 0

    self.index = 0
    self.num_point_attributes = 5
    self.lidar_paths, self.lane_paths = self.read_data(data_folder, lane_folder)
    self.frame_length = len(self.lidar_paths)
    self.lanes = self.load_lanes()
    self.points = self.load_points()

  def clusterize(self, points):
    if (self.max_clusters < 2 or points.shape[0] < self.max_clusters):
      return np.c_[points, np.ones((points.shape[0], 1))]
  
    X = points[:, self.cluster_dim].reshape(-1, 1)
    
    sil_score_max = -1 #this is the minimum possible score
    best_n_clusters = 2

    for n_clusters in range(2, self.max_clusters):
      model = KMeans(n_clusters = n_clusters, init='k-means++', max_iter=100, n_init=1)
      labels = model.fit_predict(X)
      sil_score = silhouette_score(X, labels)
      if sil_score > sil_score_max:
        sil_score_max = sil_score
        best_n_clusters = n_clusters
    
    model = KMeans(n_clusters = best_n_clusters, init='k-means++', max_iter=100, n_init=1)
    labels = model.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    print("The average silhouette score for %i clusters is %0.2f" %(best_n_clusters,sil_score))
    return np.c_[points, labels]

  def filter_points(self, raw_points):
    """Filter raw_points based on their intensity and lidar beam value"""
    filtered_points = raw_points
    if self.lidar_beam_value > -1.0:
      filtered_points = filtered_points[filtered_points[:,4] == self.lidar_beam_value]
      
    filtered_points = filtered_points[filtered_points[:,3] >= self.intensity_threshold]
    filtered_points = filtered_points[filtered_points[:,3] <= self.intensity_ceiling]
      
    filtered_points = filtered_points[filtered_points[:,0] <= self.x_ceiling]
    filtered_points = filtered_points[-filtered_points[:,0] <= self.x_ceiling]

    if(self.lanes_coefs.shape[0] > 2):
      filtered_points = filtered_points[(filtered_points[:,1] - np.polyval(self.lanes_coefs[2], filtered_points[:,0])) <= self.y_ceiling]
      filtered_points = filtered_points[-(filtered_points[:,1] - np.polyval(self.lanes_coefs[2], filtered_points[:,0])) <= self.y_ceiling]
    else:
      filtered_points = filtered_points[filtered_points[:,1] <= self.y_ceiling]
      filtered_points = filtered_points[-filtered_points[:,1] <= self.y_ceiling]
    

    return filtered_points
  
  def load_points(self):
    raw_points = np.fromfile(self.lidar_paths[self.index], dtype=np.float32).reshape(-1, self.num_point_attributes)

    filtered_points = self.filter_points(raw_points)

    filtered_points[:,3] = np.clip(filtered_points[:,3],0,51)

    filtered_points = self.clusterize(filtered_points)
    
    self.num_points = raw_points.shape[0]
    self.num_points_filtered = filtered_points.shape[0]

    xyz = filtered_points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if self.max_clusters < 2:
      colors = self.get_point_color_using_intensity(filtered_points)
    else:
      colors = self.get_point_color_using_cluster(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
  
  def load_lanes(self):
    lane_path = self.lane_paths[self.index]
    if lane_path is not None:
      lanes_coefs = []
      with open(lane_path, "r") as f:
        for line in f:
          try:
            lane_coefs = [float(x) for x in line.strip().split(";")]
          except:
            lane_coefs = [0.0,0.0,0.0,0.0]
          lanes_coefs.append(lane_coefs)
      lanes_coefs = np.array(lanes_coefs)
      self.lanes_coefs = lanes_coefs
      
      connect = []
      colors = []
      x_max = 40
      num = 80
      xs = np.linspace(start=-x_max, stop=x_max, num=num, endpoint=True)
      
      lanes = []
      for lane_coef in lanes_coefs:
        ys = np.polyval(lane_coef, xs)

        lane = np.stack([xs, ys, np.zeros_like(xs, dtype=np.float32)], axis=-1)
        lanes.append(lane)
      
      if len(lanes) > 0:
        lanes = np.concatenate(lanes, axis=0)

      nb_lanes = max(len(lanes_coefs),1)
      scale_factor = 255 // nb_lanes

      cmap = plt.get_cmap("viridis")

      # Initialize the matplotlib color map
      sm = plt.cm.ScalarMappable(cmap=cmap)

      # Obtain linear color range
      color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

      color_range = color_range.reshape(256, 3).astype(np.float32) / 255.0

      for i in range(0, nb_lanes):
        connect += [[j, j + 1] for j in range(i * len(xs), (i+1) * len(xs) - 1)]
        lane_index = np.clip(i * scale_factor, 0, 255)
        lane_index = lane_index.astype(np.uint8)
        colors += [color_range[lane_index] for _ in range(i * len(xs), (i+1) * len(xs) - 1)]
      connect = np.array(connect)
      colors = np.array(colors)

      lines = o3d.geometry.LineSet()
      lines.points = o3d.utility.Vector3dVector(lanes)
      lines.lines = o3d.utility.Vector2iVector(connect)
      lines.colors = o3d.utility.Vector3dVector(colors)
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

  def get_point_color_using_cluster(self, points):
    scale_factor = 255 // self.max_clusters
    scaled_intensity = np.clip(points[:, 5] * scale_factor, 0, 255)
    scaled_intensity = scaled_intensity.astype(np.uint8)
    cmap = plt.get_cmap("viridis")

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    color_range = color_range.reshape(256, 3).astype(np.float32) / 255.0
    colors = color_range[scaled_intensity]
    return colors

  def get_point_color_using_intensity(self, points):
    scale_factor = 5
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
            elif hasattr(key, 'char') and key.char == 'c' and self.intensity_ceiling > 0:
                self.intensity_ceiling -= 1
            elif hasattr(key, 'char') and key.char == 'd' and self.intensity_ceiling < 255:
                self.intensity_ceiling += 1
            elif hasattr(key, 'char') and key.char == 'v' and self.max_clusters > 1:
                self.max_clusters -= 1
            elif hasattr(key, 'char') and key.char == 'f' and self.max_clusters < 20:
                self.max_clusters += 1
            elif hasattr(key, 'char') and key.char == 'b' and self.cluster_dim > 0:
                self.cluster_dim -= 1
            elif hasattr(key, 'char') and key.char == 'g' and self.cluster_dim < 4:
                self.cluster_dim += 1
            elif hasattr(key, 'char') and key.char == 'z' and self.x_ceiling > 0.1:
                self.x_ceiling -= 0.1
            elif hasattr(key, 'char') and key.char == 'a':
                self.x_ceiling += 0.1
            elif hasattr(key, 'char') and key.char == 'x' and self.y_ceiling > 0.1:
                self.y_ceiling -= 0.1
            elif hasattr(key, 'char') and key.char == 's':
                self.y_ceiling += 0.1
            else:
              return
            
            self.lanes = self.load_lanes()
            self.points = self.load_points()
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
    return f"Frame Index: {self.index} / {self.frame_length - 1} - \
{self.lidar_paths[self.index]} | \
Lidar Beam Value : {self.lidar_beam_value} | \
Intensity Threshold : {self.intensity_threshold} | \
Intensity Ceiling : {self.intensity_ceiling} | \
X Ceiling : {self.x_ceiling} | \
Y Ceiling : {self.y_ceiling} | \
Max Clusters : {self.max_clusters} | \
Clusterizing Dimension : {self.cluster_dim} - \
{self.num_points_filtered} / {self.num_points} pts"

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
