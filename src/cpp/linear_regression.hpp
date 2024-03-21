#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include <iostream>
#include <Eigen/Dense>

class LinearRegression {
private:
  Eigen::MatrixXd X; // Design matrix
  Eigen::VectorXd y; // Target vector
  Eigen::VectorXd weights; // Model coefficients
  double lambda; // Ridge regularization parameter

public:
  LinearRegression(const Eigen::MatrixXd&, const Eigen::VectorXd&, double);

  // Fit the linear regression model
  void fit();

  // Predict target values for new data
  Eigen::VectorXd predict(const Eigen::MatrixXd&) const;

  // Get the coefficients (weights) of the model
  Eigen::VectorXd coefficients() const;
};

#endif /* LINEAR_REGRESSION_HPP */
