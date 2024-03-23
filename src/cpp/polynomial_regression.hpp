#ifndef POLYNOMIAL_REGRESSION_HPP
#define POLYNOMIAL_REGRESSION_HPP

#include <Eigen/Dense>
#include "linear_regression.hpp"

class PolynomialRegression {
private:
  Eigen::MatrixXd poly_X; // Polynomial inputs matrix
  int degree; // Degree of the polynomial to use
  LinearRegression linear_model; // Ridge regression model

// Generate a matrix containing the polynomial features of the input features
  Eigen::MatrixXd generatePolynomialFeatures(const Eigen::MatrixXd&) const;

public:
  PolynomialRegression(
    const Eigen::MatrixXd&,
    const Eigen::VectorXd&,
    const Eigen::VectorXd&,
    double,
    int
  );

  PolynomialRegression(double, int);

  PolynomialRegression();

  // Updates the input matrix and the target vector
  void updateData(const Eigen::MatrixXd&, const Eigen::VectorXd&, const Eigen::VectorXd&);

  // Setter for the coefficients
  void setCoefficients(const Eigen::VectorXd&);

  // Fit the polynomial regression model
  void fit();

  // Predict target values for new data
  Eigen::VectorXd predict(const Eigen::MatrixXd& new_data) const;

  // Get the distance to prediction for each row
  Eigen::VectorXd distanceToModel(const Eigen::MatrixXd&, const Eigen::VectorXd&) const;

  // Get the coefficients of the model
  Eigen::VectorXd getCoefficients() const;

  // Compute the avg of squared residuals
  double avgSquaredResiduals() const;

  // Compute the R^2 score of the model
  double score() const;
};

#endif /* POLYNOMIAL_REGRESSION_HPP */
