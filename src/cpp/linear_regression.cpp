#include "linear_regression.hpp"
#include <iostream>
#include <Eigen/Dense>

LinearRegression::LinearRegression(
  const Eigen::MatrixXd& features,
  const Eigen::VectorXd& targets,
  double lambda
): X(features.rows(), features.cols() + 1), y(targets), lambda(lambda) {
  // Add a column of ones to the features matrix for the intercept term
  X.block(0, 0, features.rows(), features.cols()) = features;
  X.col(features.cols()) = Eigen::VectorXd::Ones(features.rows());
}

// Fit the ridge regression model
void LinearRegression::fit() {
  Eigen::MatrixXd XtX = X.transpose() * X;
  Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(XtX.rows(), XtX.cols());
  weights = (XtX + lambda * eye).ldlt().solve(X.transpose() * y);
}

// Predict target values for new data
Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd& new_data) const {
  Eigen::MatrixXd new_X = Eigen::MatrixXd::Ones(new_data.rows(), new_data.cols() + 1);
  new_X.block(0, 0, new_data.rows(), new_data.cols()) = new_data;
  return new_X * weights;
}

// Get the coefficients (weights) of the model
Eigen::VectorXd LinearRegression::coefficients() const {
  return weights;
}

// Compute the R^2 score of the model
double LinearRegression::score() const {
  Eigen::VectorXd y_pred = X * weights;
  double y_mean = y.mean();
  double ss_tot = (y.array() - y_mean).square().sum();
  double ss_res = (y.array() - y_pred.array()).square().sum();
  return 1.0 - (ss_res / ss_tot);
}
