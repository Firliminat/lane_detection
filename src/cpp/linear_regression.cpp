#include "linear_regression.hpp"
#include <Eigen/Dense>

LinearRegression::LinearRegression(
  const Eigen::MatrixXd& inputs,
  const Eigen::VectorXd& targets,
  const Eigen::VectorXd& weights,
  double lambda = 0.0
):
  X(inputs.rows(), inputs.cols() + 1),
  lambda(lambda)
{
  updateData(inputs, targets, weights);
}

LinearRegression::LinearRegression(double lambda):
  X(Eigen::MatrixXd::Zero(1, 1)),
  y(Eigen::VectorXd::Zero(1)),
  weights(Eigen::VectorXd::Zero(1)),
  lambda(lambda)
{}

LinearRegression::LinearRegression():
  X(Eigen::MatrixXd::Zero(1, 1)),
  y(Eigen::VectorXd::Zero(1)),
  weights(Eigen::VectorXd::Zero(1)),
  lambda(0.0)
{}


// Updates the input matrix, the targets vector and the weights vector
void LinearRegression::updateData(
  const Eigen::MatrixXd& new_inputs,
  const Eigen::VectorXd& new_targets,
  const Eigen::VectorXd& new_weights = Eigen::VectorXd()
) {
  // Making sure inputs are not empty
  Eigen::MatrixXd inputs(new_inputs);
  if (inputs.rows() < 1) {
    inputs.conservativeResize(1, Eigen::NoChange);
  }

  // Add a column of ones to the input matrix for the intercept term
  X.resize(inputs.rows(), inputs.cols() + 1);
  X.block(0, 0, inputs.rows(), inputs.cols()) = inputs;
  X.col(inputs.cols()) = Eigen::VectorXd::Ones(inputs.rows());


  // Making sure targets are not empty
  Eigen::VectorXd targets(new_targets);
  if (targets.size() < 1) {
    targets.conservativeResize(1);
  }

  if (targets.size() != inputs.rows()) {
    throw std::invalid_argument("Invalid size of targets vector.");
  }
  y = Eigen::VectorXd(targets);


  // If weights are provided, use them; otherwise, initialize weights to ones
  if (new_weights.size() < 1) {
    weights = Eigen::VectorXd::Ones(inputs.rows());
  } else {
    if (new_weights.size() != inputs.rows()) {
      throw std::invalid_argument("Invalid size of weights vector.");
    }
    weights = Eigen::VectorXd(new_weights);
  }

  // Making sure weights are positive and weights vector is unitary norm1
  weights = Eigen::VectorXd(weights.array().abs());
  weights = Eigen::VectorXd(weights.array() / weights.sum());
}

// Setter for the coefficients
void LinearRegression::setCoefficients(const Eigen::VectorXd& new_coefficients) {
  if (new_coefficients.size() != X.cols()) {
    throw std::invalid_argument("Invalid size of coefficients vector.");
  }

  coefficients = Eigen::VectorXd(new_coefficients);
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

// Get the distance to prediction for each row
Eigen::VectorXd LinearRegression::distanceToModel(
  const Eigen::MatrixXd& inputs,
  const Eigen::VectorXd& targets
) const {
  return Eigen::VectorXd((predict(inputs).array() - targets.array()).square());
}

// Get the coefficients of the model
Eigen::VectorXd LinearRegression::getCoefficients() const {
  return coefficients;
}

// Compute the average of squared residuals
double LinearRegression::avgSquaredResiduals() const {
  Eigen::VectorXd residuals = Eigen::VectorXd(weights.array() * (y.array() - y_pred.array()));
  return residuals.squaredNorm() / residuals.size();
}

// Compute the R^2 score of the model
double LinearRegression::score() const {
  double ss_res = (weights.array() * (y.array() - y_pred.array())).square().sum();
  double ss_tot = (weights.array() * (y.array() - y.mean())).square().sum();
  return 1.0 - (ss_res / ss_tot);
}
