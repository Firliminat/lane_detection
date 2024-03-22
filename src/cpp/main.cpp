#include <vector>
#include <iostream>
#include <Eigen\Dense>
#include "data_handler.hpp"
#include "polynomial_regression.hpp"

int main(){

  DataHandler* data_handler = DataHandler::getInstance("..\\..\\pointclouds", "..\\..\\sample_output", 5);
  data_handler->parseFolders();

  Eigen::MatrixXd points = data_handler->readPoints(0);

  // Assign some example data
  Eigen::MatrixXd features = points.col(0);
  Eigen::VectorXd targets = points.col(1);
  Eigen::VectorXd weights = points.col(3);

  // Polynomial degree and regularization parameter
  int degree = 3;
  double lambda = 0.1;

  // Create PolynomialRegression object and fit the model
  PolynomialRegression model(features, targets, weights, lambda, degree);
  model.fit();

  // Output coefficients and predictions
  std::cout << "Coefficients: \n" << model.getCoefficients() << std::endl;
  
  // Compute and output the avg of squared residuals
  std::cout << "Avg of squared residuals: " << model.avgSquaredResiduals() << std::endl;

  // Compute and output the R^2 score
  std::cout << "R^2 score: " << model.score() << std::endl;

  return 0;
}