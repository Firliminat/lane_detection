#include "polynomial_regression.hpp"
#include <Eigen/Dense>
#include "linear_regression.hpp"

// Generate polynomial features matrix
Eigen::MatrixXd PolynomialRegression::generatePolynomialFeatures(const Eigen::MatrixXd& features, int degree) const {
  int num_samples = features.rows();
  int num_features = features.cols();
  Eigen::MatrixXd poly_features(num_samples, degree * num_features);
  // Eigen::MatrixXd poly_features(num_samples, degree * num_features + 1);
  // poly_features.col(0) = Eigen::VectorXd::Ones(num_samples);
  for (int i = 0; i < num_samples; ++i) {
    int index = 0;
    // int index = 1;
    for (int j = 0; j < num_features; ++j) {
      for (int d = degree; d > 0; --d) {
      // for (int d = 1; d <= degree>; ++d) {
        poly_features(i, index++) = std::pow(features(i, j), d);
      }
    }
  }
  return poly_features;
}

PolynomialRegression::PolynomialRegression(
  const Eigen::MatrixXd& features,
  const Eigen::VectorXd& targets,
  const Eigen::VectorXd& weights = Eigen::VectorXd(),
  double lambda = 0.0,
  int degree = 0
) :
  X_poly(generatePolynomialFeatures(features, degree)),
  linear_model(X_poly, targets, weights, lambda)
{}

// Fit the polynomial regression model
void PolynomialRegression::fit() {
  linear_model.fit();
}

// Predict target values for new data
Eigen::VectorXd PolynomialRegression::predict(const Eigen::MatrixXd& new_data) const {
  Eigen::MatrixXd poly_new_data = generatePolynomialFeatures(new_data, X_poly.cols() - 1);
  return linear_model.predict(poly_new_data);
}

// Get the coefficients of the model
Eigen::VectorXd PolynomialRegression::getCoefficients() const {
    return linear_model.getCoefficients();
}

// Compute the sum of squared residuals
double PolynomialRegression::sumSquaredResiduals() const {
  return linear_model.sumSquaredResiduals();
}

// Compute the R^2 score of the model
double PolynomialRegression::score() const {
  return linear_model.score();
}