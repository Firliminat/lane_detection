#include "linear_regression.hpp"
#include <Eigen/Dense>
#include "tools.hpp"

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

  // Making sure weights are between 0.0 and 1.0 and unitary
  double min_w = this->weights.minCoeff(), max_w = this->weights.maxCoeff();
  if(min_w != max_w) {
    this->weights = Eigen::VectorXd((this->weights.array() - min_w) / (max_w - min_w));
  } else {
    this->weights = Eigen::VectorXd(this->weights.array().abs());
  }
  this->weights = Eigen::VectorXd(this->weights.array() / this->weights.sum());
}

// Fit the ridge regression model
void LinearRegression::fit() {
  Eigen::MatrixXd W = weights.asDiagonal(); // Diagonal matrix of weights
  Eigen::MatrixXd XtWX = X.transpose() * W * X;
  Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(XtWX.rows(), XtWX.cols());
  coefficients = (XtWX + lambda * eye).ldlt().solve(X.transpose() * W * y);
  y_pred = X * coefficients;
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

// Compute the sum of squared residuals
double LinearRegression::sumSquaredResiduals() const {
  Eigen::VectorXd residuals = Eigen::VectorXd(weights.array() * (y.array() - y_pred.array()));
  return residuals.squaredNorm();
}

// Compute the R^2 score of the model
double LinearRegression::score() const {
  double ss_tot = (weights.array() * (y.array() - y.mean())).square().sum();
  return 1.0 - (sumSquaredResiduals() / ss_tot);
}
