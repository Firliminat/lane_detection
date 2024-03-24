#include "polynomial_regression.hpp"
#include <Eigen/Dense>
#include "linear_regression.hpp"
#include <iostream>

// Generate a matrix containing the polynomial features of the input features
Eigen::MatrixXf PolynomialRegression::generatePolynomialFeatures(
  const Eigen::MatrixXf& inputs
) const {
  if (degree < 1){
    return Eigen::MatrixXf(inputs.rows(),0);
  }

  /*
  Going through the poly_features matrix (P) to edit it with the right coefficient.
  Using this formula : P(i, j*l + l-d) = X(i, j)^d.
  With P the poly_features matrix, X the inputs matrix, 0 <= i < m, 0 <= j < n, 0 < d <= l,
  and m = X.rows() the nb of samples, n = X.cols() the nb of features, l = degree the maximum degree.
  Visually the resulting matrix is like this:
  X(0,0)^1 X(0,0)^2 ... X(0,1)^l   X(0,1)^1 ... X(0,n)^l
  X(1,0)^1 X(1,0)^2 ... X(1,0)^l   X(1,1)^1 ... X(1,n)^l
    ...                                           ...
  X(i,j)^1          ... X(i,j)^l X(i,j+1)^1 ... X(i,n)^l
    ...                                           ...
  X(m,0)^1          ... X(m,j)^l X(m,j+1)^1 ... X(m,n)^l
  */
  int num_samples = inputs.rows();
  int num_features = inputs.cols();
  Eigen::MatrixXf poly_features(num_samples, degree * num_features);
  for (int i = 0; i < num_samples; ++i) {
    int index = 0;
    for (int j = 0; j < num_features; ++j) {
      for (int d = degree; d > 0; --d) {
        poly_features(i, index++) = std::pow(inputs(i, j), d);
      }
    }
  }
  return poly_features;
}

PolynomialRegression::PolynomialRegression(
  const Eigen::MatrixXf& inputs,
  const Eigen::VectorXf& targets,
  const Eigen::VectorXf& weights = Eigen::VectorXf(),
  float lambda = 0.0,
  int degree = 0
) {
  this->degree = degree;
  poly_X = generatePolynomialFeatures(inputs);
  linear_model = LinearRegression(poly_X, targets, weights, lambda);
}

PolynomialRegression::PolynomialRegression(
  float lambda = 0.0,
  int degree = 0
) {
  this->degree = degree;
  poly_X = generatePolynomialFeatures(Eigen::MatrixXf(1, 1));
  linear_model = LinearRegression(poly_X, Eigen::VectorXf::Zero(1), Eigen::VectorXf::Ones(1), lambda);
}

PolynomialRegression::PolynomialRegression() {
  this->degree = degree;
  poly_X = generatePolynomialFeatures(Eigen::MatrixXf(1, 1));
  linear_model = LinearRegression(poly_X, Eigen::VectorXf::Zero(1), Eigen::VectorXf::Ones(1), 0.0);
}

// Updates the input matrix and the target vector
void PolynomialRegression::updateData(
  const Eigen::MatrixXf& new_inputs,
  const Eigen::VectorXf& new_targets,
  const Eigen::VectorXf& new_weights
) {
  // Making sure inputs are not empty
  Eigen::MatrixXf inputs(new_inputs);
  if (inputs.rows() < 1) {
    inputs.conservativeResize(1, Eigen::NoChange);
  }
  if (inputs.cols() < 1) {
    inputs.conservativeResize(Eigen::NoChange, 1);
  }
  
  // Making sure targets are not empty
  Eigen::VectorXf targets(new_targets);
  if (targets.size() < 1) {
    targets.conservativeResize(inputs.rows());
  }
  
  // Making sure weights are not empty
  Eigen::VectorXf weights(new_weights);
  if (weights.size() < 1) {
    weights.conservativeResize(inputs.rows());
  }
  // Making sure weights are positive
  weights = Eigen::VectorXf(weights.array().abs());
  // and weights vector is unitary for norm 1
  weights = Eigen::VectorXf(weights.array() / weights.sum());

  // Updating with the safe values
  poly_X = generatePolynomialFeatures(inputs);
  linear_model.updateData(poly_X, targets, weights);
}

// Setter for the coefficients
void PolynomialRegression::setCoefficients(const Eigen::VectorXf& new_coefficients) {
  linear_model.setCoefficients(new_coefficients);
}

// Fit the polynomial regression model
void PolynomialRegression::fit() {
  linear_model.fit();
}

// Predict target values for new data
Eigen::VectorXf PolynomialRegression::predict(const Eigen::MatrixXf& new_data) const {
  Eigen::MatrixXf poly_new_data = generatePolynomialFeatures(new_data);
  return linear_model.predict(poly_new_data);
}

// Get the squared distance to prediction for each row
Eigen::VectorXf PolynomialRegression::squaredDistancesToModel(
  const Eigen::MatrixXf& inputs,
  const Eigen::VectorXf& targets
) const {
  Eigen::MatrixXf new_X = generatePolynomialFeatures(inputs);
  return linear_model.squaredDistancesToModel(new_X, targets);
}

// Get the coefficients of the model
Eigen::VectorXf PolynomialRegression::getCoefficients() const {
    return linear_model.getCoefficients();
}

// Compute the sum of squared residuals
float PolynomialRegression::weightedSumSquaredResiduals() const {
  return linear_model.weightedSumSquaredResiduals();
}

// Compute the R^2 score of the model
float PolynomialRegression::score() const {
  return linear_model.score();
}