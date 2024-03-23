#include <iostream>
#include <Eigen\Dense>
#include "tools.hpp"
#include "data_handler.hpp"
#include "multi_polynomial_regression.hpp"

void filter(Eigen::MatrixXd points) {
  for (int point_index = 0; point_index < points.rows(); ++point_index) {
    if (points(point_index, 3) < 15) {
      Tools::removeRow(points, point_index);
    }
  }
}

void test_main() {
  int degree = 3;
  Eigen::VectorXd coefficients(degree + 1);
  coefficients.block(0, 0, degree, 1) = Eigen::VectorXd::Zero(degree);
  std::cout << coefficients << std::endl;
}

void real_main() {
  DataHandler* data_handler = DataHandler::getInstance(".\\pointclouds", ".\\sample_output", 5);
  data_handler->parseFolders();

  Eigen::MatrixXd points = data_handler->readPoints(0);

  filter(points);

  // Assign some example data
  Eigen::MatrixXd inputs = points.col(0);
  Eigen::VectorXd targets = points.col(1);
  // Eigen::VectorXd weights = points.col(3);
  Eigen::VectorXd weights = Eigen::VectorXd::Ones(inputs.rows());

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
}

int main(){
  real_main();
  return 0;
}