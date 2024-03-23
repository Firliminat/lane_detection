#include "polynomial_regression.hpp"
#include <Eigen/Dense>
#include "linear_regression.hpp"

// Generate a matrix containing the polynomial features of the input features
Eigen::MatrixXd PolynomialRegression::generatePolynomialFeatures(
  const Eigen::MatrixXd& inputs
) const {
  if (degree < 1){
    return Eigen::MatrixXd(1,0);
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
  Eigen::MatrixXd poly_features(num_samples, degree * num_features);
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
  const Eigen::MatrixXd& inputs,
  const Eigen::VectorXd& targets,
  const Eigen::VectorXd& weights = Eigen::VectorXd(),
  double lambda = 0.0,
  int degree = 0
) {
  this->degree = degree;
  this->poly_X = generatePolynomialFeatures(inputs);
  this->linear_model = LinearRegression(poly_X, Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1), lambda);
}

PolynomialRegression::PolynomialRegression(
  double lambda = 0.0,
  int degree = 0
) {
  this->degree = degree;
  this->poly_X = generatePolynomialFeatures(Eigen::MatrixXd(1, 1));
  this->linear_model = LinearRegression(poly_X, Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1), lambda);
}

PolynomialRegression::PolynomialRegression() :
  degree(0)
{
  this->poly_X = generatePolynomialFeatures(Eigen::MatrixXd(1, 1));
  this->linear_model = LinearRegression(poly_X, Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1), 0.0);
}

// Updates the input matrix and the target vector
void PolynomialRegression::updateData(
  const Eigen::MatrixXd& new_inputs,
  const Eigen::VectorXd& new_targets,
  const Eigen::VectorXd& new_weights
) {
  // Making sure inputs are not empty
  Eigen::MatrixXd inputs(new_inputs);
  if (inputs.rows() < 1) {
    inputs.conservativeResize(1, Eigen::NoChange);
  }
  if (inputs.cols() < 1) {
    inputs.conservativeResize(Eigen::NoChange, 1);
  }
  
  // Making sure targets are not empty
  Eigen::VectorXd targets(new_targets);
  if (targets.size() < 1) {
    targets.conservativeResize(1);
  }
  
  // Making sure weights are not empty
  Eigen::VectorXd weights(new_weights);
  if (weights.size() < 1) {
    weights.conservativeResize(1);
  }

  // Updating with the safe values
  poly_X = generatePolynomialFeatures(inputs);
  linear_model.updateData(poly_X, targets, weights);
}

// Setter for the coefficients
void PolynomialRegression::setCoefficients(const Eigen::VectorXd& new_coefficients) {
  linear_model.setCoefficients(new_coefficients);
}

// Fit the polynomial regression model
void PolynomialRegression::fit() {
  linear_model.fit();
}

// Predict target values for new data
Eigen::VectorXd PolynomialRegression::predict(const Eigen::MatrixXd& new_data) const {
  Eigen::MatrixXd poly_new_data = generatePolynomialFeatures(new_data);
  return linear_model.predict(poly_new_data);
}

// Get the distance to prediction for each row
Eigen::VectorXd PolynomialRegression::distanceToModel(const Eigen::MatrixXd& inputs, const Eigen::VectorXd& targets) const {
  Eigen::MatrixXd new_X = generatePolynomialFeatures(inputs);
  return linear_model.distanceToModel(new_X, targets);
}

// Get the coefficients of the model
Eigen::VectorXd PolynomialRegression::getCoefficients() const {
    return linear_model.getCoefficients();
}

// Compute the avg of squared residuals
double PolynomialRegression::avgSquaredResiduals() const {
  return linear_model.avgSquaredResiduals();
}

// Compute the R^2 score of the model
double PolynomialRegression::score() const {
  return linear_model.score();
}