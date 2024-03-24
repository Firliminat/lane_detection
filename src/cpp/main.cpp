#include <iostream>
#include <Eigen\Dense>

#include "tools.hpp"
#include "data_handler.hpp"
#include "multi_polynomial_regression.hpp"
#include "polynomial_regression.hpp"

// Filter points based on their intensity (= points(pointidx, 3))
// We keep only points such that min < intensity < max
Eigen::MatrixXf intensity_filter(
  const Eigen::MatrixXf& points,
  const int min = -1,
  const int max = 256
) {
  Eigen::MatrixXf filtered_points(0, points.cols());
  for (int pointidx = 0; pointidx < points.rows(); ++pointidx) {
    if (min < points(pointidx, 3) && points(pointidx, 3) < max) {
      filtered_points.conservativeResize(filtered_points.rows() + 1, Eigen::NoChange);
      filtered_points.row(filtered_points.rows() - 1) = points.row(pointidx);
    }
  }
  return filtered_points;
}

int test_main() {
  int num_features = 5;
  std::vector<float> input_buffer = 
    {-5, -4, -3, -2, -1,
      1,  2,  3,  4,  5,
     -5, -4, -3, -2, -1,
      1,  2,  3,  4,  5};
  int num_samples = input_buffer.size()/num_features;
  Eigen::MatrixXf inputs = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(input_buffer.data(), num_samples, num_features);
  std::cout << "inputs :\n" << inputs << std::endl;

  return 0;
}

int real_main() {
  DataHandler* data_handler = DataHandler::getInstance(".\\pointclouds", ".\\sample_output", 5);

  std::cout << "Loading data" << std::endl;
  Eigen::MatrixXf points = data_handler->readPoints(0);

  std::cout << "Filtering data" << std::endl;
  points = intensity_filter(points, 15);

  // Assign data
  Eigen::MatrixXf inputs = points.col(0);
  Eigen::VectorXf targets = points.col(1);
  Eigen::VectorXf weights = points.col(3);

  std::cout << "weights extremums: " << weights.minCoeff() << " | " << weights.maxCoeff() << std::endl;

  // Ridge regularization parameter, polynomial degree and number of models
  float lambda = 0.1;
  int degree = 3;

  std::cout << "Fitting the model to the data" << std::endl;
  MultiPolynomialRegression model(inputs, targets, weights, lambda, degree);
  model.fitNumModels(2, 10, true, true);

  Eigen::MatrixXf coefficients = model.getCoefficients();
  // Output coefficients and predictions
  std::cout << "Number of models: " << coefficients.rows() << std::endl;
  // Output coefficients and predictions
  std::cout << "Coefficients: \n" << coefficients << '\n' << std::endl;
  
  // Compute and output the avg of squared residuals
  std::cout << "Avg of squared residuals: " << model.avgSquaredResiduals() << std::endl;

  // Compute and output the R^2 score
  std::cout << "R^2 score: " << model.avgScore() << std::endl;

  data_handler->writeLanes(0, model.getCoefficients());
  
  return 0;
}

int main() {
  // return test_main();
  return real_main();
}