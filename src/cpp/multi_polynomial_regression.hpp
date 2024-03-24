#ifndef MULTI_POLYNOMIAL_REGRESSION_HPP
#define MULTI_POLYNOMIAL_REGRESSION_HPP

#include <Eigen/Dense>
#include "polynomial_regression.hpp"

class MultiPolynomialRegression {
private:
  Eigen::MatrixXf X; // Input matrix
  Eigen::VectorXf y; // Target vector
  Eigen::VectorXf w; // Weights vector
  Eigen::VectorXi a; // Indexes of the polynomials assigned to the points
  Eigen::VectorX<PolynomialRegression> poly_models; // Polynomial regression models
  float lambda; // Ridge regularization parameter
  int degree; // Degree of the polynomials
  
  Eigen::VectorXi countAssignedPoints() const;

  // Updates the given inputs, targets and weights with the ones assigned to model with given index
  void updateWithAssignedData(
    Eigen::MatrixXf&,
    Eigen::VectorXf&,
    Eigen::VectorXf&,
    const int
  ) const;

  // Assigns the data points to the closest polynomial
  void assignToModels(
    const Eigen::MatrixXf& inputs,
    const Eigen::VectorXf& targets,
    const Eigen::VectorXf& weights,
    Eigen::VectorXi& a
  ) const;

  // Initialize the polynomial models
  void initModels(const int);

  // fit the polynomial models to the data points assigned to them
  void fitModels();

public:
  MultiPolynomialRegression(
    const Eigen::MatrixXf& = Eigen::MatrixXf::Zero(1, 1),
    const Eigen::VectorXf& = Eigen::VectorXf::Zero(1),
    const Eigen::VectorXf& = Eigen::VectorXf(),
    const float = 0.0,
    const int = 0,
    const int = 1
  );

  // Fit the model
  void fit();

  // Fit the number of models
  void fitNumModels(const int = 1, const int = 20);

  // Predict target values for new data for all models
  Eigen::MatrixXf predict(const Eigen::MatrixXf& new_data) const;

  // Get the squared distances to prediction for each model
  Eigen::MatrixXf squaredDistancesToModels(
    const Eigen::MatrixXf& inputs,
    const Eigen::VectorXf& targets,
    const Eigen::VectorXf& weights
  ) const;

  // Get the squared distance to prediction for each row
    Eigen::VectorXf squaredDistancesToModel(
      const Eigen::MatrixXf&,
      const Eigen::VectorXf&,
      const Eigen::VectorXf&,
      const Eigen::VectorXi& = Eigen::VectorXi()
    ) const;

  // Get the coefficients of the models
  Eigen::MatrixXf getCoefficients() const;

  // Compute the the sum squared residuals to assigned model
  float sumSquaredResiduals() const;

  // Compute the average model wise of the sum squared residuals
  float avgSquaredResiduals() const;

  // Compute the average model wise of the R^2 score
  float avgScore() const;
};

#endif /* MULTI_POLYNOMIAL_REGRESSION_HPP */
