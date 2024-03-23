import threading
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from sklearn import preprocessing
from pynput import keyboard

class Visualization():
  """Class used to visualize data"""

  def __init__(
    self,
    title: str='Visualization',
    points=np.zeros((0,4), dtype=np.float32),
    lanes_coefs=np.zeros((0,4), dtype=np.float32)
  ):
    """Visualization constructor:
    Inputs:
      title: title of the visualization window.
      points: points to display. points[0:3] are the xyz dimensions and points[3:6] are used for colorization
      lanes: array of coefs of the polynomials representing tyhe lanes"""
    
    self.title = title
    """Title of the visualization window"""
    self.points = points
    """Data to visualize"""
    self.lanes_coefs = lanes_coefs
    """Polynomials coefficients of the lanes to visualize"""

    self._is_running = False
    """Boolean stating if the o3D app is running"""
    self._3d = None
    """Open3D scene"""
    self._window = None
    """Display window"""

    self._event: threading.Event = None
    """Event used to update or stop api running thread"""

    self._api_run_thread: threading.Thread = None
    """Thread running the api"""
    self._update_from_input_thread: threading.Thread = None
    """Thread checking keyboard inputs"""
    self._update_or_stop_from_event_thread: threading.Thread = None
    """Thread checking _update_or_stop event"""

  def run_or_update(
    self,
    title = 'Visualization',
    points = np.zeros((0,6)),
    lanes_coefs = []
  ):
    self._update(title, points, lanes_coefs)
    if not self._is_running:
      self.run()
    else:
      self.update(title, points, lanes_coefs)
    return self

  def run(self):
    """Starts the visualization window"""

    if self._is_running:
      return
    self._is_running = True

    self._event = threading.Event()
    def init_and_run(vis):
      thread_vis = Visualization(
        vis.title,
        vis.points, 
        vis.lanes_coefs
      )
      thread_vis._event = vis._event
      thread_vis._run()
    
    self._api_run_thread = threading.Thread(target=init_and_run, args=[self], daemon=True)
    self._api_run_thread.start()
  
  def update(self, title=None, points=None, lanes_coefs=None):
    """Updates the visualization window"""

    self._update(title, points, lanes_coefs)
    self._event.stop = False
    self._event.title = self.title
    self._event.points = self.points
    self._event.lanes_coefs = self.lanes_coefs
    self._event.set()

  def stop(self):
    """Stops the visualization window"""

    self._event.stop = True
    self._event.set()
    self._is_running = False


  def update_title(self, new_title: str=None):
    """Updates the title"""

    if new_title is not None:
      self.title = new_title
      self._update_window_title()


  def update_points(self, new_points=None):
    """Updates the points"""

    if new_points is not None:
      self.points = new_points
      self._update_points_geometry()
  

  def update_lanes_coefs(self, new_lanes_coefs=None):
    """Updates the lanes coefs"""

    if new_lanes_coefs is not None:
      self.lanes_coefs = new_lanes_coefs
      self._update_lanes_geometry()

  def _run(self):
    """Initiates Open3D app and starts it"""

    if self._is_running:
      return
    self._is_running = True

    # Initiating the o3d app
    gui.Application.instance.initialize()
    self._window = gui.Application.instance.create_window(self.title, 1280, 720)
    self._3d = gui.SceneWidget()
    self._3d.scene = rendering.Open3DScene(self._window.renderer)
    self._3d.scene.set_background([0.3, 0.3, 0.3, 1.0])
    self._window.add_child(self._3d)

    # Updating the geometry using the data
    self._update_points_geometry()
    self._update_lanes_geometry()

    # Setting up listeners
    self._update_from_input()
    self._update_from_event()

    # Running the app
    gui.Application.instance.run()


  def _update(
    self,
    new_title: str=None,
    new_points=None,
    new_lanes_coefs=None
  ):
    """Updates the display using the given new values"""

    self.update_title(new_title)
    self.update_points(new_points)
    self.update_lanes_coefs(new_lanes_coefs)

    if self._window is not None:
      self._window.post_redraw()


  def _stop(self):
    """Quits the o3d application"""

    gui.Application.instance.quit()
    self._is_running = False
    self._window = None
    self._3d = None


  def _update_from_input(self):
    """Creates a thread listening for keyboard inputs"""

    def update_from_input_thread():
      """Creates a keyboard listener with the wished callback"""

      def on_press(key):
        """Updating the visualization when f5 is pressed. Method to call when a key is pressed"""

        if key == keyboard.Key.f5:
          gui.Application.instance.post_to_main_thread(
            self._window,
            self._update
          )

      with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    self._update_from_input_thread = threading.Thread(target=update_from_input_thread, daemon=True)
    self._update_from_input_thread.start()

  def _update_from_event(self):
    """Creates a thread listening for _event"""

    def update_wapper(vis):
      stop = vis._event.stop
      title = vis._event.title
      points = vis._event.points
      lanes_coefs = vis._event.lanes_coefs
      vis._event.stop = None
      vis._event.title = None
      vis._event.points = None
      vis._event.lanes_coefs = None
      def update():
        if stop:
          vis._stop()
        else:
          vis._update(title, points, lanes_coefs)
      return update

    def update_from_event_thread(vis):
      vis._event.wait()
      vis._event.clear()
      vis._update_from_event()
      gui.Application.instance.post_to_main_thread(
        vis._window,
        update_wapper(vis)
      )

    self._update_or_stop_from_event_thread = threading.Thread(target=update_from_event_thread, args=[self], daemon=True)
    self._update_or_stop_from_event_thread.start()

  
  def _update_window_title(self):
    """Updates the title using self.title"""

    if self._window is None:
      return
    
    self._window.title = self.title


  def _update_points_geometry(self):
    """Updates the points geometry using self.points"""

    if self._3d is None or self.points is None:
      return

    xyz = self.points[:, :3]
    self._pcd = o3d.geometry.PointCloud()
    self._pcd.points = o3d.utility.Vector3dVector(xyz)

    colors = self._get_point_color_using_last_dimensions()
    self._pcd.colors = o3d.utility.Vector3dVector(colors)
    name = "__points__"
    if self._3d.scene.has_geometry(name):
        self._3d.scene.remove_geometry(name)
    self._3d.scene.add_geometry(name, self._pcd, rendering.MaterialRecord())

    # Updates the camera now that we have updated self._pcd
    self._update_camera()

  def _get_point_color_using_last_dimensions(self):
    """Transforms last self.points dimensions into colors."""

    color_values = self.points[:, 3:6]
    if color_values.shape[1] < 1:
       color_values = np.ones((self.points.shape[0], 3))
    for _ in range(3 - color_values.shape[1]):
      color_values = np.c_[color_values, color_values[:,0]]
    scaler = preprocessing.MinMaxScaler().fit(color_values)
    scaled_colors = scaler.transform(color_values).astype(np.float32)
    return scaled_colors

  def _update_lanes_geometry(self):
    """Updates the lanes geometry using self.lanes_coefs"""

    if self._3d is None or self.lanes_coefs is None:
      return
    
    connect = []
    colors = []
    x_max = 40
    num = 80
    xs = np.linspace(start=-x_max, stop=x_max, num=num, endpoint=True)
    
    lanes = []
    for lane_coef in self.lanes_coefs:
      ys = np.polyval(lane_coef, xs)

      lane = np.stack([xs, ys, np.zeros_like(xs, dtype=np.float32)], axis=-1)
      lanes.append(lane)
    
    if len(lanes) > 0:
      lanes = np.concatenate(lanes, axis=0)

    nb_lanes = max(len(self.lanes_coefs),1)
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

    self._lanes = o3d.geometry.LineSet()
    self._lanes.points = o3d.utility.Vector3dVector(lanes)
    self._lanes.lines = o3d.utility.Vector2iVector(connect)
    self._lanes.colors = o3d.utility.Vector3dVector(colors)
    
    name = "__lanes__"
    if self._3d.scene.scene.has_geometry(name):
      self._3d.scene.remove_geometry(name)
    if self._lanes is not None:
      mat = rendering.MaterialRecord()
      mat.shader = "unlitLine"
      mat.line_width = 2 * self._window.scaling
      self._3d.scene.add_geometry(name, self._lanes, mat)

  def _update_camera(self):
    """Updates the camera"""

    if(self._pcd is None or self._3d is None):
      return
    
    # Setting up the camera
    bounds = self._pcd.get_axis_aligned_bounding_box()
    self._3d.setup_camera(60, bounds, bounds.get_center())
