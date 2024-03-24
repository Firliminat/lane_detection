#ifndef POLYNOMIAL_REGRESSION_HPP
#define POLYNOMIAL_REGRESSION_HPP

#include <Eigen/Dense>
#include "linear_regression.hpp"

class PolynomialRegression {
private:
  Eigen::MatrixXf poly_X; // Polynomial inputs matrix
  int degree; // Degree of the polynomial to use
  LinearRegression linear_model; // Ridge regression model

// Generate a matrix containing the polynomial features of the input features
  Eigen::MatrixXf generatePolynomialFeatures(const Eigen::MatrixXf&) const;

public:
  PolynomialRegression(
    const Eigen::MatrixXf&,
    const Eigen::VectorXf&,
    const Eigen::VectorXf&,
    float,
    int
  );

  PolynomialRegression(float, int);

  PolynomialRegression();

  // Updates the input matrix and the target vector
  void updateData(const Eigen::MatrixXf&, const Eigen::VectorXf&, const Eigen::VectorXf&);

  // Setter for the coefficients
  void setCoefficients(const Eigen::VectorXf&);

  // Fit the polynomial regression model
  void fit();

  // Predict target values for new data
  Eigen::VectorXf predict(const Eigen::MatrixXf& new_data) const;

  // Get the squared distance to prediction for each row
    Eigen::VectorXf squaredDistancesToModel(
      const Eigen::MatrixXf&,
      const Eigen::VectorXf&,
      const Eigen::VectorXf&
    ) const;

  // Get the coefficients of the model
  Eigen::VectorXf getCoefficients() const;

  // Compute the sum of squared residuals
  float sumSquaredResiduals() const;

  // Compute the R^2 score of the model
  float score() const;
};

#endif /* POLYNOMIAL_REGRESSION_HPP */
