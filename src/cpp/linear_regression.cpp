#include "linear_regression.hpp"
#include <Eigen/Dense>
#include <iostream>

LinearRegression::LinearRegression(
  const Eigen::MatrixXf& inputs,
  const Eigen::VectorXf& targets,
  const Eigen::VectorXf& weights,
  float lambda = 0.0
) {
  X = Eigen::MatrixXf(inputs.rows(), inputs.cols() + 1);
  this->lambda = lambda;
  updateData(inputs, targets, weights);
}

LinearRegression::LinearRegression(float lambda):
  X(Eigen::MatrixXf::Zero(1, 1)),
  y(Eigen::VectorXf::Zero(1)),
  w(Eigen::VectorXf::Zero(1)),
  lambda(lambda)
{}

LinearRegression::LinearRegression():
  X(Eigen::MatrixXf::Zero(1, 1)),
  y(Eigen::VectorXf::Zero(1)),
  w(Eigen::VectorXf::Zero(1)),
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
    w = Eigen::VectorXf::Ones(inputs.rows());
  } else {
    if (new_weights.size() != inputs.rows()) {
      throw std::invalid_argument("Invalid size of weights vector.");
    }
    w = Eigen::VectorXf(new_weights);
  }
  // Making sure weights vector is unitary for norm 1
  float w_sum = w.sum();
  if(w_sum == 0.0) {
    throw std::invalid_argument("Norm 1 of weights is 0.0");
  }
  w = Eigen::VectorXf(w.array() / w.sum());
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
  Eigen::MatrixXf W = w.asDiagonal(); // Diagonal matrix of weights
  Eigen::MatrixXf XtWX = X.transpose() * W * X;
  Eigen::MatrixXf eye = Eigen::MatrixXf::Identity(XtWX.rows(), XtWX.cols());
  coefficients = (XtWX + lambda * eye).ldlt().solve(X.transpose() * W * y);
}

// Predict target values for new data
Eigen::VectorXf LinearRegression::predict(const Eigen::MatrixXf& new_data) const {
  Eigen::MatrixXf new_X(new_data);

  // Making sure new data has the good number of features
  int new_num_features = new_X.cols();
  int num_features = X.cols();
  if (new_num_features < X.cols()) {
    int num_features_to_add = num_features - new_num_features;
    new_X.conservativeResize(Eigen::NoChange, num_features);
    new_X.block(0, new_num_features, new_X.rows(), num_features_to_add) = Eigen::MatrixXf::Ones(new_X.rows(), num_features_to_add);
  }

  // Making the predictions
  return new_X * coefficients;
}

// Get the distance to prediction for each row
Eigen::VectorXf LinearRegression::distancesToModel(
  const Eigen::MatrixXf& inputs,
  const Eigen::VectorXf& targets
) const {
  return Eigen::VectorXf(targets.array() - predict(inputs).array());
}

// Get the squared distance to prediction for each row
Eigen::VectorXf LinearRegression::squaredDistancesToModel(
  const Eigen::MatrixXf& inputs,
  const Eigen::VectorXf& targets
) const {
  return Eigen::VectorXf(distancesToModel(inputs, targets).array().square());
}

// Get the coefficients of the model
Eigen::VectorXf LinearRegression::getCoefficients() const {
  return coefficients;
}

// Compute the sum of squared residuals
float LinearRegression::weightedSumSquaredResiduals() const {
  Eigen::VectorXf residuals = Eigen::VectorXf(w.array() * squaredDistancesToModel(X, y).array());
  return residuals.sum();
}

// Compute the score of the model
float LinearRegression::score() const {
  float ss_res = weightedSumSquaredResiduals();
  float ss_tot = (w.array() * (y.array() - y.mean()).square()).sum();
  return 1.0 - (ss_res / ss_tot);
}
