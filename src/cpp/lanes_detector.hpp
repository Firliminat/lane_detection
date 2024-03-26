#ifndef LANES_DETECTOR_HPP
#define LANES_DETECTOR_HPP

#include <Eigen/Dense>

#include "multi_polynomial_regression.hpp"
#include "polynomial_regression.hpp"

// Class used for detecting the two closest lanes
class LanesDetector {
private:
  const Eigen::Index X_COL = 0; // Col-index for x-values
  const Eigen::Index Y_COL = 1; // Col-index for y-values
  const Eigen::Index Z_COL = 2; // Col-index for z-values
  const Eigen::Index I_COL = 3; // Col-index for intensity values
  const Eigen::Index B_COL = 4; // Col-index for beam index values
  const Eigen::Index D_COL = 5; // Col-index for squared distances to road profile

  Eigen::MatrixXf inputs; // Lidar points

  Eigen::MatrixXf t_inputs; // Lidar points on the top of the road profile
  Eigen::MatrixXf b_inputs; // Lidar points on the bottom of the road profile

  int degree; // Degree of the polynomials used to fit to the lanes
  float max_lane_width; // Max width of a road lane
  float min_lane_width; // Min width of a road lane
  float lane_width_delta; // How much to decrease the lane_width on each iteration

  float max_intensity; // Maximum intensity used for intensity clipping
  float min_intensity; // Maximum intensity used for intensity filtering

  float lambda_road_profile; // Ridge regularization parameter for road profiling
  float lambda_lanes; // Ridge regularization parameter for lanes

  PolynomialRegression road_profile; // Polynomial regression model used to estimate the profile of the road
  PolynomialRegression t_lane; // Polynomial regression model used to estimate the top lane
  PolynomialRegression b_lane; // Polynomial regression model used to estimate the bottom lane

  // Applies the preprocessing
  void preProcess();

  // Fit the road profile to all the points and centers it
  void computeRoadProfile();

  // Remove the points too far from the road profile
  void filterFarthestPoints(const float, const float = 0.0);

  // Remove the points too far from the road profile
  // and split between botom ones and top ones
  void splitFilterPoints(const float, const float = 0.0);
  
  // Update distances to road profile
  void updateDistances(const float =  0.0);

  // Find lanes in the remaining points
  void findLanes();

public:
  LanesDetector(
    const Eigen::MatrixXf& = Eigen::MatrixXf::Zero(1, 5), // inputs
    const int = 3, // degree
    const float = 10.0, // max_lane_width
    const float = 0.5, // min_lane_width
    const float = 0.1, // lane_width_delta
    const float = 255.0, // max_intensity
    const float = 0.0, // min_intensity
    const float = 0.0, // lambda_road_profile
    const float = 0.0 // lambda_lanes
  );

  // Fit the model
  void fit(bool = false);

  
  // Get the coefficients of the lanes
  Eigen::MatrixXf getLanesCoefficients() const;

  // Get the R^2 scores the lanes in this order : min, max, avg
  Eigen::VectorXf silhouetteScores() const;

  // Get the R^2 scores the lanes in this order : min, max, avg
  Eigen::VectorXf scores() const;

  // Get the min R^2 scores of the lanes
  float score() const;

};

#endif /* LANES_DETECTOR_HPP */
