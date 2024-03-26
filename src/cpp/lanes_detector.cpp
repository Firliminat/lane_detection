#include "lanes_detector.hpp"

#include <Eigen/Dense>

#include "multi_polynomial_regression.hpp"
#include "polynomial_regression.hpp"
#include "tools.hpp"

// Applies the preprocessing
void LanesDetector::preProcess() {
  // Removing inputs with low intensity
  inputs = Tools::rowWiseFilter(inputs, I_COL, min_intensity);

  // Clipping the intensity values to avoid points being too heavy
  inputs.col(I_COL) = Tools::clip(inputs.col(I_COL), 0.0, max_intensity);
}

// Remove the points too far from the road profile
void LanesDetector::filterFarthestPoints(const float lane_width) {
  inputs = Tools::rowWiseFilter(inputs, D_COL, -lane_width, lane_width);
}

// Update distances to road profile
void LanesDetector::updateDistances() {
  inputs.col(D_COL) = road_profile.distancesToModel(
    inputs.col(X_COL),
    inputs.col(Y_COL)
  );
}

// Fit the road profile to all the points and centers it
void LanesDetector::computeRoadProfile() {
  road_profile = PolynomialRegression(
    degree,
    lambda_road_profile, 
    inputs.col(X_COL),
    inputs.col(Y_COL),
    inputs.col(I_COL)
  );
  road_profile.fit();

  // Center the profile by setting the x^0 term to 0.0
  Eigen::VectorXf road_coefs = road_profile.getCoefficients();
  road_coefs(road_coefs.size() - 1) = 0.0;
  road_profile.setCoefficients(road_coefs);
}

// Fit the road profile to all the points and centers it
void LanesDetector::findLanes() {
  Eigen::VectorXf coefficients = Eigen::VectorXf::Zero(degree + 1);
  coefficients(degree - 1) = road_profile.getCoefficients()(0);
  lanes = MultiPolynomialRegression(
    inputs.col(X_COL),
    inputs.col(Y_COL),
    inputs.col(I_COL),
    lambda_lanes,
    degree,
    // road_profile.getCoefficients(),
    coefficients,
    inputs.col(D_COL).minCoeff(),
    inputs.col(D_COL).maxCoeff()
  );
  lanes.fitNumModels(min_num_lanes, max_num_lanes, min_improvement);
}


LanesDetector::LanesDetector(
  const Eigen::MatrixXf& inputs,
  const int degree,
  const float max_lane_width,
  const float min_lane_width,
  const float lane_width_delta,
  const float max_intensity,
  const float min_intensity,
  const int min_num_lanes,
  const int max_num_lanes,
  const float min_improvement,
  const float lambda_road_profile,
  const float lambda_lanes
) :
  degree(degree),
  max_lane_width(max_lane_width),
  min_lane_width(min_lane_width),
  lane_width_delta(lane_width_delta),
  max_intensity(max_intensity),
  min_intensity(min_intensity),
  min_num_lanes(min_num_lanes),
  max_num_lanes(max_num_lanes),
  min_improvement(min_improvement),
  lambda_road_profile(lambda_road_profile),
  lambda_lanes(lambda_lanes)
  {
    // Adding a column to inputs for squared distance to road profile
    this->inputs = inputs;
    this->inputs.conservativeResize(Eigen::NoChange, 6);
  }


// Fit the model
void LanesDetector::fit(bool verbose) {

  preProcess();

  Eigen::MatrixXf inputs_save(inputs);

  if(verbose) {
    Tools::printTitles({
      "iter",
      "lane width",
      "best width",
      "avg silh",
      "min silh",
      "max silh",
      "avg R^2",
      "sumResiduals",
      "nbPoints"
    }, 13);
  }

  float
    best_lane_width = max_lane_width,
    lane_width = max_lane_width,
    score = -1;
  int iter_idx = 0;
  while (lane_width >= min_lane_width && ++iter_idx < 100) {

    computeRoadProfile();
    updateDistances();
    filterFarthestPoints(lane_width);
    findLanes();
    
    Eigen::MatrixXf silh_scores = lanes.weightedSilhouetteScores();
    float new_score = silh_scores.maxCoeff();
    if(new_score > score) {
      best_lane_width = lane_width;
      score = new_score;
    }
    if(verbose) {
      float avg_silh = silh_scores.mean();
      float min_silh = silh_scores.minCoeff();
      Tools::printRow({
        static_cast<float>(iter_idx),
        lane_width,
        best_lane_width,
        avg_silh,
        min_silh,
        new_score,
        lanes.avgScore(),
        lanes.weightedSumSquaredResiduals(),
        static_cast<float>(inputs.rows())
      }, 13);
    }

    lane_width -= lane_width_delta;
  }

  // We fit with the best paramter
  if (best_lane_width > min_lane_width) {
    inputs = inputs_save;

    computeRoadProfile();
    updateDistances();
    filterFarthestPoints(best_lane_width);
    findLanes();
  }
}

// Get the coefficients of the lanes
Eigen::MatrixXf LanesDetector::getLanesCoefficients() const {
  Eigen::MatrixXf coefficients(lanes.getCoefficients());
  coefficients.conservativeResize(coefficients.rows() + 1, Eigen::NoChange);
  coefficients.row(coefficients.rows() - 1) = road_profile.getCoefficients();
  return coefficients;
}
