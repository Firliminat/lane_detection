#ifndef MULTI_POLYNOMIAL_REGRESSION_HPP
#define MULTI_POLYNOMIAL_REGRESSION_HPP

#include <Eigen/Dense>
#include "polynomial_regression.hpp"

class MultiPolynomialRegression {
private:
  Eigen::MatrixXd X; // Input matrix
  Eigen::VectorXd y; // Target vector
  Eigen::VectorXd weights; // Weights vector
  Eigen::VectorXi points_assignments; // Indexes of the polynomials assigned to the points
  Eigen::VectorX<PolynomialRegression> poly_models; // Polynomial regression models
  double lambda; // Ridge regularization parameter
  int degree; // Degree of the polynomials
  
  Eigen::VectorXi countAssignedPoints() const;

  // Updates the given inputs, targets and weights with the ones assigned to model with given index
  void updateWithAssignedData(
    Eigen::MatrixXd&,
    Eigen::VectorXd&,
    Eigen::VectorXd&,
    int
  ) const;

  // Assigns the data points tothe closest polynomial
  void assignToPolynomials();

  // Initialize the polynomial models
  void initModels(int);

  // fit the polynomial models to the data points assigned to them
  void fitModels();

public:
  MultiPolynomialRegression(
    const Eigen::MatrixXd&,
    const Eigen::VectorXd&,
    const Eigen::VectorXd&,
    double,
    int,
    int
  );

  // Fit the model
  void fit();

  // Predict target values for new data for all models
  Eigen::MatrixXd predict(const Eigen::MatrixXd& new_data) const;

  // Get the coefficients of the models
  Eigen::MatrixXd getCoefficients() const;

  // Compute the avg of squared residuals
  double avgSquaredResiduals() const;

  // Compute the average R^2 score of the models
  double avgScore() const;
};

#endif /* MULTI_POLYNOMIAL_REGRESSION_HPP */
