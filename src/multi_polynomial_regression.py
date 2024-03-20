from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class MultiPolynomialRegression():
  """Class used for clusterizing using multiple univariate polynomial regressions"""

  def __init__(self, min_num_polynomials=2, max_num_polynomials=20, max_fit_iterations=100, min_delta=15, min_epsilon=5, deg=3, points=np.zeros((0,2))):
    """MultiPolynomialRegression constructor:
    Inputs:
      min_num_polynomials: minimal number of polynomials to fit to the points.
      max_num_polynomials: maximal number of polynomials to fit to the points.
      max_fit_iterations: maximal number of iterations to fit the polynomials the points.
      min_delta: minimum score diminution when growing the polynomial family.
      min_epsilon: minimum score diminution when fitting the polynomials.
      num_polynomials: number of polynomials to fit to the points.
      deg: degree of the polynomials.
      points: data to fit the polynomial to. Format of the points : (inputs, targets)"""
    self.num_polynomials = max(min_num_polynomials, 1)
    self.max_num_polynomials = max(max_num_polynomials, 1)
    self.max_fit_iterations = max_fit_iterations
    self.min_delta = min_delta
    self.min_epsilon = min_epsilon
    self.deg = deg
    self.points = points[:,0:2]
    
    self.score = np.inf
    self.polynomial_family = MultiPolynomialRegression.null_polynomial_family(self.num_polynomials, self.deg)
    self.scores = np.zeros((0,2))


  @staticmethod
  def null_polynomial_family(num_polynomials=1, deg=3) -> np.ndarray:
    """Returns a family of num_polynomials null polynomials of the given degree"""

    return np.array([PolynomialRegression.null_polynomial(deg).tolist() for _ in range(num_polynomials)])

  @staticmethod
  def dist_to_polynomial(point=np.zeros((2)), polynomial=np.zeros(1)) -> float:
    """Returns an estimate of the distance between the given point and polynomial.
    Inputs:
      point: Point in [x, y] format.
      polynomial: Coefficients of the polynomial in descendig order.
    Output: Height between the point and the polynomial squared, not the real distance but a simple and good estimate."""

    return np.square(point[1] - np.polyval(polynomial, point[0]))
  
  def compute_score(self) -> None:
    """Compute current score of the regression.
    Behavior:
      Iterate over each point to add it's distance to it's assigned polynomial to the score.
      Then divides it by the number of polynomials to get the averaged distance.
    Updates: self.score"""

    # If negative degree or 0 polynoms or points are not assigned yet we return an infinite score
    if self.deg < 0 or self.num_polynomials < 1 or self.points.shape[1] < 3:
      self.score = np.inf
      return
    
    score = 0.0
    for point_index, point in enumerate(self.points):
      try:
        assigned_polynomial_index = min(int(point[2]), self.num_polynomials - 1)
        assigned_polynomial = self.polynomial_family[assigned_polynomial_index]
        score += MultiPolynomialRegression.dist_to_polynomial(point, assigned_polynomial)
      except:
        print(f'Couldn\'t compute cost for point {point_index}: {point}')
        print(f'Current polynomial_family: {self.polynomial_family}')
    self.score = score / self.num_polynomials

  def plot_score_evolution(self):
    _, ax = plt.subplots()
    ax.plot(self.scores[:,0], self.scores[:,1])

    ax.set(xlabel='Number of polynomials', ylabel='Score',
          title='Evolution of the score')
    ax.grid()

    plt.show()

  def init_polynomial_family(self):
    """Initalizes the polynomial family.
    Behavior:
      Create a family of constant polynoms uniformally spread allong the points Y axis 
    Updates: self.polynomial_family"""

    # If no points or negative degree or 0 polynoms allowed we can't init the polynoms
    if self.points.shape[0] < 1 or self.deg < 0 or self.num_polynomials < 1:
      return
    
    min_y = min(self.points[:,1])
    delta_y = max(self.points[:,1]) - min_y
    self.polynomial_family = np.array([
        [0.0, 0.0, 0.0, (i + 1) * delta_y / (self.num_polynomials + 1) + min_y]
        for i in range(self.num_polynomials)
      ])

  def assign_points(self) -> None:
    """Assign each point to the closest polynomial.
    Behavior:
      For each point iterates over the polynomial_family to assign the closest one
    Updates: self.points"""

    # If no points or negative degree or 0 polynoms allowed we can't assign the polynoms
    if self.points.shape[0] < 1 or self.num_polynomials < 1:
      return
    
    if self.points.shape[1] < 3:
      self.points = np.c_[self.points, np.zeros(self.points.shape[0])]

    for point in self.points:
      min_dist = np.inf
      for poly_index, polynomial in enumerate(self.polynomial_family):
        new_dist = MultiPolynomialRegression.dist_to_polynomial(point, polynomial)
        if new_dist < min_dist:
          min_dist = new_dist
          point[2] = poly_index

  def fit_polynomials(self) -> None:
    """Fits each polynomial of current self.polynomial_family to the points assigned to it.
    Behavior:
      Iterates over the polynomial family to fit each polynomial to the assigned points.
    Updates: self.polynomial_family"""

    # If no points or negative degree or 0 polynoms allowed or the points are not assigned yet we can't fit the polynomials
    if self.points.shape[0] < 1 or self.deg < 0 or self.num_polynomials < 1 or self.points.shape[1] < 3:
      self.polynomial_family = MultiPolynomialRegression.null_polynomial_family(self.num_polynomials, self.deg)
      return
    
    for poly_index in range(self.num_polynomials):
      assigned_points = self.points[self.points[:,2] == poly_index]
      model = PolynomialRegression(self.deg, assigned_points)
      model.fit()
      self.polynomial_family[poly_index] = model.polynomial


  def fit_polynomial_family(self, polynomials_fitting_callback: Callable=None, verbose=False) -> None:
    """Fits the family of self.num_polynomials polynomials to self.points.
    Inputs:
      polynomials_fitting_callback: ((MultiPolynomialRegression) -> Any) callback used during the fitting loop.
      verbose: Boolean stating if we should print what's going on
    Behavior:
      Combines clustering and polynomial regression to fit a family of polynomials to the points.
    Updates: self.score, self.points, self.polynomial_family"""

    # If no points or negative degree or 0 polynoms allowed we can't fit polynomials
    if self.points.shape[0] < 1 or self.deg < 0 or self.num_polynomials < 1:
      self.polynomial_family = MultiPolynomialRegression.null_polynomial_family(self.num_polynomials, self.deg)
      return
    
    if verbose:
      print(f'\tInitializing {self.num_polynomials} polynomials')
    self.init_polynomial_family()
    self.compute_score()

    if verbose:
      print(f'\tFitting polynomials to the points')
    
    # Initializing lopping conditions.
    # Fitting stops when the scores doesn't improove more than min_score_delta
    # or when we reach the maximum number of iterations
    score_condition = True
    loop_index = 0
    while score_condition and loop_index < self.max_fit_iterations:
      old_score = self.score

      self.assign_points()
      self.fit_polynomials()
      if polynomials_fitting_callback is not None:
        polynomials_fitting_callback(self)

      # Updating score condition
      self.compute_score()
      score_delta = old_score - self.score
      score_condition = score_delta > self.min_epsilon

      if verbose:
        print(f'\t\t{loop_index}: {self.score} | {score_delta}')
        print(f'\t\t{self.num_polynomials}: ' + " | ".join([f'{len(self.points[self.points[:, 2] ==  poly_index])}' for poly_index in range(self.num_polynomials)]))

      loop_index += 1
    

  def fit(
    self,
    score_plotting: bool=False,
    size_fitting_callback: Callable=None,
    polynomials_fitting_callback: Callable=None,
    verbose=False,
    talkative=False
  ):
    """Fits a family of polynomials to self.points. Also fits the size of the family.
    Inputs:
      size_fitting_callback: ((MultiPolynomialRegression) -> Any) callback used during the polynomial family size fitting loop.
      polynomials_fitting_callback: ((MultiPolynomialRegression) -> Any) callback used during the polynomial family fitting loop.
      verbose: Boolean stating if fit_polynomial_family should print what's going on
      talkative: Boolean stating if fit_polynomials should print what's going on
    Behavior:
      Combines clustering and polynomial regression to fit a family of polynomials to the points.'
      Keeps growing the polynomial family untill scores doesn't improve enough
    Updates: self.score, self.points, self.polynomial_family, self.num_polynomials"""

    # If no points or negative degree we can't fit a polynomial family
    if self.points.shape[0] < 1 or self.deg < 0:
      self.polynomial_family = MultiPolynomialRegression.null_polynomial_family(self.num_polynomials, self.deg)
      return

    def iterate(
      model: MultiPolynomialRegression,
      size_fitting_callback: Callable=None,
      polynomials_fitting_callback: Callable=None,
      verbose=False,
      talkative=False
    ) -> bool:
      """Represents one iteration of the polynomial family size fitting algortihm.
      Inputs:
        model: the current model we are using
        size_fitting_callback: ((MultiPolynomialRegression) -> Any) callback used during the polynomial family size fitting loop.
        polynomials_fitting_callback: ((MultiPolynomialRegression) -> Any) callback used during the polynomial family fitting loop.
        verbose: Boolean stating if fit_polynomial_family should print what's going on
        talkative: Boolean stating if fit_polynomials should print what's going on
      Behavior:
        Fits a polynomial family of self.num_polynomials to the points. Updates the score accordignly.
        Calls the size fitting callback and prints stpes if verbose is True
      Updates: self.score, self.points, self.polynomial_family, self.num_polynomials"""

      old_score = model.score

      model.fit_polynomial_family(polynomials_fitting_callback, verbose=talkative)
      if size_fitting_callback is not None:
        size_fitting_callback(model)

      # Updating score
      model.compute_score()
      score_delta = old_score - model.score

      if verbose:
        print(f'\t{model.num_polynomials}: {model.score} | {score_delta}')
      
      return score_delta

    if verbose:
      print(f'Fitting polynomial family to the points')
    
    # Initializing lopping conditions.
    # Growing the polynomial family stops when the score doesn't improove more than min_score_delta
    # or when we reach the maximum number of iterations
    score_condition = True
    score_delta = np.inf
    while score_condition and self.num_polynomials < self.max_num_polynomials:
      score_delta = iterate(
        self,
        size_fitting_callback,
        polynomials_fitting_callback,
        verbose,
        talkative
      )
      self.scores = np.append(self.scores, [[self.num_polynomials, self.score]], axis=0)
      
      # Updating lopping conditions.
      score_condition = score_delta > self.min_delta
      self.num_polynomials += 1
    # Compensating for end of final loop incrementation
    self.num_polynomials -= 1
    
    # We go back ot the previous family size
    # if the final iteration actually decreased the score
    if score_delta < 0:
      self.num_polynomials -= 1
      score_delta = iterate(
        self,
        size_fitting_callback,
        polynomials_fitting_callback,
        verbose,
        talkative
      )
    
    if(score_plotting):
      self.plot_score_evolution()



class PolynomialRegression():
  """Class used for univariate polynomial regression"""

  def __init__(self, deg=3, points=np.zeros((0,2))):
    """PolynomialRegression constructor:
    Inputs:
      deg: degree of the polynomial.
      points: data to fit the polynomial to. Format of the points : (inputs, targets)"""

    self.deg = deg
    self.points = points[:,0:2]
    self.polynomial = PolynomialRegression.null_polynomial(self.deg)

  @staticmethod
  def null_polynomial(deg=3) -> np.ndarray:
    """Returns a null polynomial of the given degree"""

    return np.array([0.0 for _ in range(deg + 1)])

  def fit(self) -> None:
    """Fits a polynomial to self.points.
    Behavior:
      Creates polynomial features based on the input and use linear regression on these features
    Updates: self.polynomial"""

    # If no points or negative degree we can't fit
    if self.points.shape[0] < 1 or self.deg < 0:
      self.polynomial = PolynomialRegression.null_polynomial(self.deg)
      return

    # Creating the polynomial features from inputs
    poly = PolynomialFeatures(degree=self.deg, include_bias=False)
    poly_features = poly.fit_transform(self.points[:,0].reshape(-1, 1))
    
    # Creating and fitting a polynomial to the points
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, self.points[:,1])
    self.polynomial = np.append(np.flip(poly_reg_model.coef_), [poly_reg_model.intercept_])

