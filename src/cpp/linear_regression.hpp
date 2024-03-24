#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include <Eigen/Dense>

class LinearRegression {
private:
  Eigen::MatrixXf X; // Input matrix
  Eigen::VectorXf y; // Target vector
  Eigen::VectorXf w; // Weights used to favorise some points
  Eigen::VectorXf coefficients; // Model coefficients
  float lambda; // Ridge regularization parameter

public:
  LinearRegression(
    const Eigen::MatrixXf&,
    const Eigen::VectorXf&,
    const Eigen::VectorXf&,
    float
  );

  LinearRegression(float);

  LinearRegression();

  // Updates the input matrix, the targets vector and the weights vector
  void updateData(
    const Eigen::MatrixXf&,
    const Eigen::VectorXf&,
    const Eigen::VectorXf&
  );

  // Setter for the coefficients
  void setCoefficients(const Eigen::VectorXf&);

  // Fit the linear regression model
  void fit();

  // Predict target values for new data
  Eigen::VectorXf predict(const Eigen::MatrixXf&) const;

  // Get the squared distance to prediction for each row
    Eigen::VectorXf squaredDistancesToModel(
      const Eigen::MatrixXf&,
      const Eigen::VectorXf&
    ) const;

  // Get the coefficients of the model
  Eigen::VectorXf getCoefficients() const;

  // Compute the sum of squared residuals
  float weightedSumSquaredResiduals() const;
  
  // Compute the R^2 score of the model
  float score() const;
};

#endif /* LINEAR_REGRESSION_HPP */
