#include <iostream>
#include <Eigen\Dense>
#include "tools.hpp"
#include "data_handler.hpp"
#include "multi_polynomial_regression.hpp"
#include "polynomial_regression.hpp"

void filter(Eigen::MatrixXd points) {
  for (int point_index = 0; point_index < points.rows(); ++point_index) {
    if (points(point_index, 3) < 15) {
      Tools::removeRow(points, point_index);
    }
  }
}

int test_main() {
  // Example usage
  Eigen::MatrixXd inputs(22, 1); // 5 samples, 1 feature
  Eigen::VectorXd targets(22); // Corresponding target values
  Eigen::VectorXd weights = Eigen::VectorXd::Ones(inputs.rows());

  // Assign some example data
  inputs <<   -5,  -4,  -3,  -2,  -1,  0,   1,   2,   3,   4,   5,   -5,   -4,   -3,   -2,   -1,   0,    1,    2,    3,    4,    5;
  targets << 0.5, 0.6, 0.7, 0.8, 0.9, 10, 1.1, 1.2, 1.3, 1.4, 1.5, -1.5, -1.4, -1.3, -1.2, -1.1, -10, -0.9, -0.8, -0.7, -0.6, -0.5;
  weights(5) = 0.0;
  weights(16) = 1;

  // Ridge regularization parameter, polynomial degree and number of models
  // double lambda = 0.1;
  double lambda = 0.0;
  int degree = 3;
  int num_models = 2;

  // Create PolynomialRegression object and fit the model
  MultiPolynomialRegression model(inputs, targets, weights, lambda, degree, num_models);
  model.fit();

  // Compute and output the avg of squared residuals and R^2 score
  std::cout << "Avg of squared residuals: " << model.avgSquaredResiduals() << std::endl;
  std::cout << "avg R^2 score: " << model.avgScore() << std::endl;
  std::cout << "Coeffs: \n" << std::scientific << std::setprecision(3) << model.getCoefficients() << std::endl;

  return 0;
}

int real_main() {
  DataHandler* data_handler = DataHandler::getInstance(".\\pointclouds", ".\\sample_output", 5);
  data_handler->parseFolders();

  Eigen::MatrixXd points = data_handler->readPoints(0);

  // filter(points);

  // Assign data
  Eigen::MatrixXd inputs = points.col(0);
  Eigen::VectorXd targets = points.col(1);
  Eigen::VectorXd weights = points.col(3);

  // Ridge regularization parameter, polynomial degree and number of models
  // double lambda = 0.1;
  double lambda = 0.0;
  int degree = 3;
  int num_models = 5;

  // Create PolynomialRegression object and fit the model
  MultiPolynomialRegression model(inputs, targets, weights, lambda, degree, num_models);
  model.fit();

  // Output coefficients and predictions
  std::cout << "Coefficients: \n" << model.getCoefficients() << std::endl;
  
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