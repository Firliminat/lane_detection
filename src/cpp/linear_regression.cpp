#include "linear_regression.hpp"
#include <Eigen/Dense>

LinearRegression::LinearRegression(
  const Eigen::MatrixXf& inputs,
  const Eigen::VectorXf& targets,
  const Eigen::VectorXf& weights,
  float lambda = 0.0
):
  X(inputs.rows(), inputs.cols() + 1),
  lambda(lambda)
{
  updateData(inputs, targets, weights);
}

LinearRegression::LinearRegression(float lambda):
  X(Eigen::MatrixXf::Zero(1, 1)),
  y(Eigen::VectorXf::Zero(1)),
  weights(Eigen::VectorXf::Zero(1)),
  lambda(lambda)
{}

LinearRegression::LinearRegression():
  X(Eigen::MatrixXf::Zero(1, 1)),
  y(Eigen::VectorXf::Zero(1)),
  weights(Eigen::VectorXf::Zero(1)),
  lambda(0.0)
{}


// Updates the input matrix, the targets vector and the weights vector
void LinearRegression::updateData(
  const Eigen::MatrixXf& new_inputs,
  const Eigen::VectorXf& new_targets,
  const Eigen::VectorXf& new_weights = Eigen::VectorXf()
) {
  // Making sure inputs are not empty
  Eigen::MatrixXf inputs(new_inputs);
  if (inputs.rows() < 1) {
    inputs.conservativeResize(1, Eigen::NoChange);
  }

  // Add a column of ones to the input matrix for the intercept term
  X.resize(inputs.rows(), inputs.cols() + 1);
  X.block(0, 0, inputs.rows(), inputs.cols()) = inputs;
  X.col(inputs.cols()) = Eigen::VectorXf::Ones(inputs.rows());


  // Making sure targets are not empty
  Eigen::VectorXf targets(new_targets);
  if (targets.size() < 1) {
    targets.conservativeResize(1);
  }

  if (targets.size() != inputs.rows()) {
    throw std::invalid_argument("Invalid size of targets vector.");
  }
  y = Eigen::VectorXf(targets);


  // If weights are provided, use them; otherwise, initialize weights to ones
  if (new_weights.size() < 1) {
    weights = Eigen::VectorXf::Ones(inputs.rows());
  } else {
    if (new_weights.size() != inputs.rows()) {
      throw std::invalid_argument("Invalid size of weights vector.");
    }
    weights = Eigen::VectorXf(new_weights);
  }

  // Making sure weights are positive and weights vector is unitary norm1
  weights = Eigen::VectorXf(weights.array().abs());
  weights = Eigen::VectorXf(weights.array() / weights.sum());
}

// Setter for the coefficients
void LinearRegression::setCoefficients(const Eigen::VectorXf& new_coefficients) {
  if (new_coefficients.size() != X.cols()) {
    throw std::invalid_argument("Invalid size of coefficients vector.");
  }

  coefficients = Eigen::VectorXf(new_coefficients);
}

// Fit the ridge regression model
void LinearRegression::fit() {
  Eigen::MatrixXf W = weights.asDiagonal(); // Diagonal matrix of weights
  Eigen::MatrixXf XtWX = X.transpose() * W * X;
  Eigen::MatrixXf eye = Eigen::MatrixXf::Identity(XtWX.rows(), XtWX.cols());
  coefficients = (XtWX + lambda * eye).ldlt().solve(X.transpose() * W * y);
  y_pred = X * coefficients;
}

// Predict target values for new data
Eigen::VectorXf LinearRegression::predict(const Eigen::MatrixXf& new_data) const {
  Eigen::MatrixXf new_X = Eigen::MatrixXf::Ones(new_data.rows(), new_data.cols() + 1);
  new_X.block(0, 0, new_data.rows(), new_data.cols()) = new_data;
  return new_X * coefficients;
}

// Get the distance to prediction for each row
Eigen::VectorXf LinearRegression::distanceToModel(
  const Eigen::MatrixXf& inputs,
  const Eigen::VectorXf& targets
) const {
  return Eigen::VectorXf((predict(inputs).array() - targets.array()).square());
}

// Get the coefficients of the model
Eigen::VectorXf LinearRegression::getCoefficients() const {
  return coefficients;
}

// Compute the sum of squared residuals
float LinearRegression::sumSquaredResiduals() const {
  Eigen::VectorXf residuals = Eigen::VectorXf(weights.array() * (y.array() - y_pred.array()));
  return residuals.squaredNorm();
}

// Compute the R^2 score of the model
float LinearRegression::score() const {
  float ss_res = (weights.array() * (y.array() - y_pred.array())).square().sum();
  float ss_tot = (weights.array() * (y.array() - y.mean())).square().sum();
  return 1.0 - (ss_res / ss_tot);
}
