#include "polynomial_regression.hpp"
#include <Eigen/Dense>
#include "linear_regression.hpp"

// Generate polynomial features matrix
Eigen::MatrixXd PolynomialRegression::generatePolynomialFeatures(const Eigen::MatrixXd& features, int degree) const {
  /*
  Going through the poly features matrix (P) to edit it with the right coefficient.
  Using this formula : P(i, j*l + l-d) = X(i, j)^d.
  With 0 <= i < m; 0 <= j < n; 0 < d <= l, X = features,
  and m = X.rows() the nb of samples, n = X.cols() the nb of features, l = degree the maximum degree.
  Visually the resulting matrix is like this:
  X(0,0)^1 X(0,0)^2 ... X(0,1)^l   X(0,1)^1 ... X(0,n)^l
  X(1,0)^1 X(1,0)^2 ... X(1,0)^l   X(1,1)^1 ... X(1,n)^l
    ...                                           ...
  X(i,j)^1          ... X(i,j)^l X(i,j+1)^1 ... X(i,n)^l
    ...                                           ...
  X(m,0)^1          ... X(m,j)^l X(m,j+1)^1 ... X(m,n)^l
  */
  int num_samples = features.rows();
  int num_features = features.cols();
  Eigen::MatrixXd poly_features(num_samples, degree * num_features);
  for (int i = 0; i < num_samples; ++i) {
    int index = 0;
    for (int j = 0; j < num_features; ++j) {
      for (int d = degree; d > 0; --d) {
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

// Compute the avg of squared residuals
double PolynomialRegression::avgSquaredResiduals() const {
  return linear_model.avgSquaredResiduals();
}

// Compute the R^2 score of the model
double PolynomialRegression::score() const {
  return linear_model.score();
}