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

  Eigen::VectorXf init_coefficients; // Coefficients used toinitialize the models
  float min_init_constant; // min y_value to initialize the models
  float max_init_constant; // max y_value to initialize the models

  // Updates the given inputs, targets and weights with the ones assigned to model with given index
  void updateWithAssignedData(
    Eigen::MatrixXf&,
    Eigen::VectorXf&,
    Eigen::VectorXf&, // TODO: Put weights at the end
    const int
  ) const;

  // Assigns the data points to the closest polynomial
  void assignToModels(Eigen::VectorXi& a) const;

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
    const Eigen::VectorXf = Eigen::VectorXf(),
    const float = -1.0,
    const float = 1.0
  );

  // Fit the model
  void fit(const float = 0.01, bool = false);

  // Fit the number of models
  void fitNumModels(
    const int = 1,
    const int = 20,
    const float = 0.01,
    bool = false,
    bool = false
  );

  // Predict target values for new data for all models
  Eigen::MatrixXf predict(const Eigen::MatrixXf& new_data) const;

  // Get the squared distances to prediction for each model
  Eigen::MatrixXf squaredDistancesToModels(
    const Eigen::MatrixXf& inputs,
    const Eigen::VectorXf& targets
  ) const;

  // Get the squared distance to prediction for each row
    Eigen::VectorXf squaredDistancesToModel(
      const Eigen::MatrixXf&,
      const Eigen::VectorXf&,
      const Eigen::VectorXi& = Eigen::VectorXi()
    ) const;

  // Get the coefficients of the models
  Eigen::MatrixXf getCoefficients() const;

  // Compute the the sum squared residuals to assigned model
  float weightedSumSquaredResiduals() const;

  // Compute the average model wise of the sum squared residuals
  float avgSquaredResiduals() const;

  // Compute the average model wise of the R^2 score
  float avgScore() const;

  // Computes the simplified silhouette score of the model
  float weightedSilhouetteScore() const;

  // Computes the simplified silhouette score for each model
  Eigen::VectorXf weightedSilhouetteScores() const;
  
  // Counts the number of points assigned to each model
  Eigen::VectorXi countAssignedSamples() const;

  // Sums the weights of points assigned to each model
  Eigen::VectorXf sumAssignedWeigths() const;
};

#endif /* MULTI_POLYNOMIAL_REGRESSION_HPP */
