#include <iostream>
#include <algorithm>
#include <Eigen\Dense>

#include "tools.hpp"
#include "data_handler.hpp"
#include "multi_polynomial_regression.hpp"
#include "polynomial_regression.hpp"
#include "lanes_detector.hpp"

// Filter points based on their intensity (= points(pointidx, 3))
// We keep only points such that min <= intensity <= max
Eigen::MatrixXf intensity_filter(
  const Eigen::MatrixXf& points,
  const int min = -1,
  const int max = 256
) {
  Eigen::MatrixXf filtered_points(0, points.cols());
  for (int pointidx = 0; pointidx < points.rows(); ++pointidx) {
    if (min <= points(pointidx, 3) && points(pointidx, 3) <= max) {
      filtered_points.conservativeResize(filtered_points.rows() + 1, Eigen::NoChange);
      filtered_points.row(filtered_points.rows() - 1) = points.row(pointidx);
    }
  }
  return filtered_points;
}

template<typename Derived>
typename Derived::Scalar median( Eigen::DenseBase<Derived>& d ){
    auto r { d.reshaped() };
    std::sort( r.begin(), r.end() );
    return r.size() % 2 == 0 ?
        r.segment( (r.size()-2)/2, 2 ).mean() :
        r( r.size()/2 );
}

template<typename Derived>
typename Derived::Scalar median( const Eigen::DenseBase<Derived>& d ){
    typename Derived::PlainObject m { d.replicate(1,1) };
    return median(m);
}

// Applies the ReLU function to the inputs
Eigen::VectorXf clip(const Eigen::VectorXf& inputs, const float min = 0.0, const float max = 255.0) {
  Eigen::VectorXf outputs(inputs.size());
  for (int input_idx = 0; input_idx < inputs.size(); ++input_idx) {
    outputs(input_idx) = std::min(std::max(min, inputs(input_idx)), max);
  }
  return outputs;
}

// Applies the logistic function to the inputs
Eigen::VectorXf logistic(const Eigen::VectorXf& inputs, const float x0, const float k = 1.0) {
  return Eigen::VectorXf(1.0 / (1.0 + (-k * (inputs.array() - x0)).exp() ));
}

// Applies the ReLU function to the inputs
Eigen::VectorXf reLU(const Eigen::VectorXf& inputs) {
  return Eigen::VectorXf((inputs.array() + inputs.array().abs()) / 2);
}

int real_main() {
  int num_features = 5;
  DataHandler* data_handler = DataHandler::getInstance(".\\pointclouds", ".\\sample_output", num_features);

  // Model Parameters
  int degree = 3;

  float max_lane_width = 4.0;
  float min_lane_width = 1.5;
  float lane_width_delta = 0.1;
  float min_num_lanes = 2;
  float max_num_lanes = 2;

  float max_intensity = 30.0;
  float min_intensity = 10.0;

  float min_improvement = 0.05;
  float lambda_road_profile = 0.0;
  float lambda_lanes = 0.0;

  for (int frame_idx = 4; frame_idx < 5; ++frame_idx) {
    std::cout << "Frame: " << frame_idx << std::endl;
    std::cout << "Loading data" << std::endl;
    Eigen::MatrixXf points = data_handler->readPoints(frame_idx);

    std::cout << "Fitting the model to the data" << std::endl;
    LanesDetector lanes_detector(
      points,
      degree,
      max_lane_width,
      min_lane_width,
      lane_width_delta,
      max_intensity,
      min_intensity,
      min_num_lanes,
      max_num_lanes,
      min_improvement,
      lambda_road_profile,
      lambda_lanes
    );
    lanes_detector.fit(true);

    data_handler->writeLanes(frame_idx, lanes_detector.getLanesCoefficients());
  }
  
  return 0;
}

int old_real_main() {
  DataHandler* data_handler = DataHandler::getInstance(".\\pointclouds", ".\\sample_output", 5);

  int max_degree = 3;
  // Model Parameters
  float lambda = 0.0;
  int degree = 3;
  int min_intensity = 3;
  int max_intensity = 256;
  int min_clip_intensity = 0;
  int max_clip_intensity = 10;
  float logistic_k = 2.0;
  float min_improvement = 0.5;
  float min_fit_improvement = 0.03;

  for (int frame_idx = 0; frame_idx < data_handler->num_frames; ++frame_idx) {
    std::cout << "Frame: " << frame_idx << std::endl;
    std::cout << "Loading data" << std::endl;
    Eigen::MatrixXf points = data_handler->readPoints(frame_idx);

    std::cout << "Filtering data" << std::endl;
    points = intensity_filter(points, min_intensity, max_intensity);

    // Assign data
    Eigen::MatrixXf inputs = points.col(0);
    Eigen::VectorXf targets = points.col(1);
    Eigen::VectorXf weights = points.col(3);

    weights = clip(weights, min_clip_intensity, max_clip_intensity);

    std::cout << "Fitting the model to the data" << std::endl;
    MultiPolynomialRegression model(inputs, targets, weights, lambda, degree);
    model.fitNumModels(
      1,
      1,
      min_improvement,
      true,
      false
    );

    data_handler->writeLanes(frame_idx, model.getCoefficients());
  }
  
  return 0;
}

int main() {
  Eigen::setNbThreads(8);
  // return test_main();
  return real_main();
}