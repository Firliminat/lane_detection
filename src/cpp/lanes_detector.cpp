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
void LanesDetector::filterFarthestPoints(const float lane_width, const float lane_center) {
  inputs = Tools::rowWiseFilter(inputs, D_COL, lane_center-lane_width, lane_center+lane_width);
}

// Remove the points too far from the road profile
// and split between botom ones and top ones
void LanesDetector::splitFilterPoints(const float lane_width, const float lane_center) {
  t_inputs = Tools::rowWiseFilter(inputs, D_COL, lane_center, lane_center + lane_width / 2.0);
  b_inputs = Tools::rowWiseFilter(inputs, D_COL, lane_center - lane_width / 2.0, lane_center);
  inputs.resize(t_inputs.rows() + b_inputs.rows(), Eigen::NoChange);
  inputs << t_inputs,
            b_inputs;
}

// Update distances to road profile
void LanesDetector::updateDistances(const float lane_center) {
  inputs.col(D_COL) = Eigen::VectorXf(road_profile.distancesToModel(
    inputs.col(X_COL),
    inputs.col(Y_COL)
  ).array() - lane_center);
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
  road_coefs(road_coefs.size() - 1) = 0;
  road_profile.setCoefficients(road_coefs);
}

// Fit the road profile to all the points and centers it
void LanesDetector::findLanes() {
  t_lane = PolynomialRegression(
    degree,
    lambda_lanes,
    t_inputs.col(X_COL),
    t_inputs.col(Y_COL),
    t_inputs.col(I_COL)
  );
  t_lane.fit();

  b_lane = PolynomialRegression(
    degree,
    lambda_lanes,
    b_inputs.col(X_COL),
    b_inputs.col(Y_COL),
    b_inputs.col(I_COL)
  );
  b_lane.fit();
}


LanesDetector::LanesDetector(
  const Eigen::MatrixXf& inputs,
  const int degree,
  const float max_lane_width,
  const float min_lane_width,
  const float lane_width_delta,
  const float max_intensity,
  const float min_intensity,
  const float lambda_road_profile,
  const float lambda_lanes
) :
  degree(degree),
  max_lane_width(max_lane_width),
  min_lane_width(min_lane_width),
  lane_width_delta(lane_width_delta),
  max_intensity(max_intensity),
  min_intensity(min_intensity),
  lambda_road_profile(lambda_road_profile),
  lambda_lanes(lambda_lanes)
  {
    // Adding a column to inputs for distance to road profile
    this->inputs = inputs;
    this->inputs.conservativeResize(Eigen::NoChange, 6);
    this->inputs.col(D_COL) = this->inputs.col(Y_COL);
  }


// Fit the model
void LanesDetector::fit(bool verbose) {

  preProcess();
  computeRoadProfile();

  Eigen::MatrixXf inputs_save(inputs);

  if(verbose) {
    Tools::printTitles({
      "iter",
      "lane width",
      "lane center",
      "best width",
      "best center",
      "avg silh",
      "min score",
      "top sum W",
      "bot sum W",
      "nbPoints",
    }, 13);
  }

  Eigen::MatrixXf t_coefs(0, degree + 1);
  Eigen::MatrixXf b_coefs(0, degree + 1);
  float
    lane_width = max_lane_width,
    best_lane_width = max_lane_width,
    best_lane_center = -lane_width / 2.0,
    lane_center_delta = lane_width_delta / 2.0,
    score = -1;
  int 
    iter_idx = 0,
    best_iter_idx = 0;
  while (lane_width >= min_lane_width && iter_idx < 300) {
    float lane_center = -lane_width / 2.0;

    inputs = inputs_save;
    while (lane_center <= lane_width / 2.0) {
  
      updateDistances(lane_center);
      splitFilterPoints(lane_width, lane_center);
      findLanes();

      t_coefs.conservativeResize(t_coefs.rows() + 1, Eigen::NoChange);
      t_coefs.row(t_coefs.rows() - 1) = t_lane.getCoefficients();
      b_coefs.conservativeResize(b_coefs.rows() + 1, Eigen::NoChange);
      b_coefs.row(b_coefs.rows() - 1) = b_lane.getCoefficients();
      
      float new_score = this->score();
      if(new_score > score) {
        best_lane_width = lane_width;
        best_lane_center = lane_center;
        best_iter_idx = iter_idx;
        score = new_score;
      }
      if(verbose) {
        Eigen::VectorXf silhouette_scores = silhouetteScores();
        Eigen::VectorXf scores = this->scores();

        Tools::printRow({
          static_cast<float>(iter_idx),
          lane_width,
          lane_center,
          best_lane_width,
          best_lane_center,
          silhouette_scores(0),
          scores(0),
          t_inputs.col(I_COL).sum(),
          b_inputs.col(I_COL).sum(),
          t_inputs.col(I_COL).sum() + b_inputs.col(I_COL).sum()
        }, 13);
      }
      ++iter_idx;

      lane_center += lane_center_delta;
    }

    lane_width -= lane_width_delta;
  }

  // We fit with the best paramter
  if (best_iter_idx != iter_idx) {
    t_lane.setCoefficients(t_coefs.row(best_iter_idx));
    b_lane.setCoefficients(b_coefs.row(best_iter_idx));
  }
}

// Get the coefficients of the lanes
Eigen::MatrixXf LanesDetector::getLanesCoefficients() const {
  Eigen::MatrixXf coefficients = Eigen::MatrixXf::Zero(2, degree+1);

  coefficients.row(0) = t_lane.getCoefficients();
  coefficients.row(1) = b_lane.getCoefficients();

  return coefficients;
}

// Get the silhouette scores the lanes in this order : min, max, avg
Eigen::VectorXf LanesDetector::silhouetteScores() const {

  Eigen::VectorXf t_dists_to_t = t_lane.squaredDistancesToModel(t_inputs.col(X_COL), t_inputs.col(Y_COL));
  Eigen::VectorXf t_dists_to_b = b_lane.squaredDistancesToModel(t_inputs.col(X_COL), t_inputs.col(Y_COL));
  Eigen::VectorXf b_dists_to_t = t_lane.squaredDistancesToModel(b_inputs.col(X_COL), b_inputs.col(Y_COL));
  Eigen::VectorXf b_dists_to_b = b_lane.squaredDistancesToModel(b_inputs.col(X_COL), b_inputs.col(Y_COL));

  float t_sum_weigths = t_inputs.col(I_COL).sum();
  float t_silh = 0;
  for (int sample_idx = 0; sample_idx < t_inputs.rows(); sample_idx++) {
    float sample_score = (t_dists_to_b(sample_idx) - t_dists_to_t(sample_idx)) / std::max(t_dists_to_b(sample_idx), t_dists_to_t(sample_idx));
    t_silh += t_inputs(sample_idx, I_COL) * sample_score;
  }
  t_silh = t_silh / t_sum_weigths;

  float b_sum_weigths = b_inputs.col(I_COL).sum();
  float b_silh = 0;
  for (int sample_idx = 0; sample_idx < b_inputs.rows(); sample_idx++) {
    float sample_score = (b_dists_to_t(sample_idx) - b_dists_to_b(sample_idx)) / std::max(b_dists_to_t(sample_idx), b_dists_to_b(sample_idx));
    b_silh += b_inputs(sample_idx, I_COL) * sample_score;
  }
  b_silh = b_silh / b_sum_weigths;

  Eigen::VectorXf scores = Eigen::VectorXf::Zero(3);
  scores(0) = std::min(t_silh, b_silh);
  scores(1) = std::max(t_silh, b_silh);
  scores(2) = (t_sum_weigths * t_silh + b_sum_weigths * b_silh) / (t_sum_weigths + b_sum_weigths);

  return scores;
}

// Get the R2 scores of the lanes in this order : min, max, avg
Eigen::VectorXf LanesDetector::scores() const {
  float t_score = t_lane.score();
  float b_score = b_lane.score();

  float t_sum_weigths = t_inputs.col(I_COL).sum();
  float b_sum_weigths = b_inputs.col(I_COL).sum();

  Eigen::VectorXf scores = Eigen::VectorXf::Zero(3);
  scores(0) = std::min(t_score, b_score);
  scores(1) = std::max(t_score, b_score);
  scores(2) = (t_sum_weigths * t_score + b_sum_weigths * b_score) / (t_sum_weigths + b_sum_weigths);

  return scores;
}

// Get the min R2 scores of the lanes
float LanesDetector::score() const {
  return scores()(0);
}
