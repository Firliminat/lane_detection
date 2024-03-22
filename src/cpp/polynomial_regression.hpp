#ifndef POLYNOMIAL_REGRESSION_HPP
#define POLYNOMIAL_REGRESSION_HPP

#include <Eigen/Dense>
#include "linear_regression.hpp"

class PolynomialRegression {
private:
  Eigen::MatrixXd X_poly; // Polynomial features matrix
  LinearRegression linear_model; // Ridge regression model

  // Generate polynomial features matrix
  Eigen::MatrixXd generatePolynomialFeatures(const Eigen::MatrixXd&, int) const;

public:
  PolynomialRegression(
    const Eigen::MatrixXd&,
    const Eigen::VectorXd&,
    const Eigen::VectorXd&,
    double,
    int
  );

  // Fit the polynomial regression model
  void fit();
  // Predict target values for new data
  Eigen::VectorXd predict(const Eigen::MatrixXd& new_data) const;
  // Get the coefficients of the model
  Eigen::VectorXd getCoefficients() const;
  // Compute the sum of squared residuals
  double sumSquaredResiduals() const;
  // Compute the R^2 score of the model
  double score() const;
};

#endif /* POLYNOMIAL_REGRESSION_HPP */
