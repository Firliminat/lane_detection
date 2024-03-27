#include <iostream>
#include <Eigen\Dense>

#include "tools.hpp"
#include "data_handler.hpp"
#include "lanes_detector.hpp"

// Uses several polynomial regressions to clusterize the points in to the different lanes
void nPolynomialRegressions() {
  const std::string data_folder = ".\\pointclouds";
  const std::string lanes_folder = ".\\sample_output_nPoly";
  int num_features = 5;
  
  DataHandler* data_handler = DataHandler::getInstance(data_folder, lanes_folder, num_features);
  
  for (int frame_idx = 0; frame_idx < data_handler->num_frames; ++frame_idx) {
    std::cout << "Frame: " << frame_idx << std::endl;
    std::cout << "Loading data" << std::endl;
    Eigen::MatrixXf points = data_handler->readPoints(frame_idx);

    std::cout << "Filtering data" << std::endl;
    points = Tools::rowWiseFilter(points, 3, 10.0);

    // Assign data
    Eigen::MatrixXf inputs = points.col(0);
    Eigen::VectorXf targets = points.col(1);
    Eigen::VectorXf weights = points.col(3);

    std::cout << "weights extremums: " << weights.minCoeff() << " | " << weights.maxCoeff() << std::endl;

    // Ridge regularization parameter, polynomial degree and number of models
    float lambda = 0.01;
    int degree = 3;

    std::cout << "Fitting the model to the data" << std::endl;
    MultiPolynomialRegression model(inputs, targets, weights, lambda, degree);
    model.fitNumModels(3, 8, 0.05, true, false);

    data_handler->writeLanes(frame_idx, model.getCoefficients());
  }
}

// Applies the regression to all of the frames and outputs the lanes coefficients in the corresponding folders
void twoPolynomialRegressions(){
  const std::string data_folder = ".\\pointclouds";
  const std::string lanes_folder = ".\\sample_output";
  int num_features = 5;

  DataHandler* data_handler = DataHandler::getInstance(data_folder, lanes_folder, num_features);

  // Model Parameters
  int degree = 3;

  float max_lane_width = 3.0;
  float min_lane_width = 1.8;
  float lane_width_delta = 0.05;

  float max_intensity = 20.0;
  float min_intensity = 3.0;

  float lambda_road_profile = 1.0;
  float lambda_lanes = 0.0;

  for (int frame_idx = 0; frame_idx < data_handler->num_frames; ++frame_idx) {

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
      lambda_road_profile,
      lambda_lanes
    );
    lanes_detector.fit(true);

    data_handler->writeLanes(frame_idx, lanes_detector.getLanesCoefficients());
  }
}


int main(int argc, char* argv[]) {

  std::string arg;
  if (argc > 1) {
    arg = argv[1];
  }

  if(arg == "--nPoly") {
    nPolynomialRegressions();
  } else {
    twoPolynomialRegressions();
  }

  return 0;
}