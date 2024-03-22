#include "linear_regression.hpp"
#include <iostream>
#include <Eigen/Dense>

LinearRegression::LinearRegression(
  const Eigen::MatrixXd& features,
  const Eigen::VectorXd& targets,
  const Eigen::VectorXd& weights = Eigen::VectorXd(),
  double lambda = 0.0
): X(features.rows(), features.cols() + 1), y(targets), lambda(lambda) {
  // Add a column of ones to the features matrix for the intercept term
  X.block(0, 0, features.rows(), features.cols()) = features;
  X.col(features.cols()) = Eigen::VectorXd::Ones(features.rows());

  // If weights are provided, use them; otherwise, initialize weights to ones
  if (weights.size() == 0) {
    this->weights = Eigen::VectorXd::Ones(features.rows());
  } else {
    if (weights.size() != features.rows()) {
      throw std::invalid_argument("Invalid size of weights vector.");
    }
    this->weights = weights;
  }
}

// Fit the ridge regression model
void LinearRegression::fit() {
  Eigen::MatrixXd W = weights.asDiagonal(); // Diagonal matrix of weights
  Eigen::MatrixXd XtWX = X.transpose() * W * X;
  Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(XtWX.rows(), XtWX.cols());
  coefficients = (XtWX + lambda * eye).ldlt().solve(X.transpose() * W * y);
}

// Predict target values for new data
Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd& new_data) const {
  Eigen::MatrixXd new_X = Eigen::MatrixXd::Ones(new_data.rows(), new_data.cols() + 1);
  new_X.block(0, 0, new_data.rows(), new_data.cols()) = new_data;
  return new_X * coefficients;
}

// Get the coefficients of the model
Eigen::VectorXd LinearRegression::getCoefficients() const {
  return coefficients;
}

// Compute the R^2 score of the model
double LinearRegression::score() const {
  Eigen::VectorXd y_pred = X * coefficients;
  double y_mean = y.mean();
  double ss_tot = (weights.array() * (y.array() - y_mean)).square().sum();
  double ss_res = (weights.array() * (y.array() - y_pred.array())).square().sum();
  return 1.0 - (ss_res / ss_tot);
}
