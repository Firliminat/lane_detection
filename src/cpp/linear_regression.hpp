#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include <Eigen/Dense>

class LinearRegression {
private:
  Eigen::MatrixXd X; // Input matrix
  Eigen::VectorXd y; // Target vector
  Eigen::VectorXd y_pred; // Predicted target vector
  Eigen::VectorXd coefficients; // Model coefficients
  Eigen::VectorXd weights; // Weights used to favorise some points
  double lambda; // Ridge regularization parameter

public:
  LinearRegression(
    const Eigen::MatrixXd&,
    const Eigen::VectorXd&,
    const Eigen::VectorXd&,
    double
  );

  LinearRegression(double);

  LinearRegression();

  // Updates the input matrix, the targets vector and the weights vector
  void updateData(const Eigen::MatrixXd&, const Eigen::VectorXd&, const Eigen::VectorXd&);

  // Setter for the coefficients
  void setCoefficients(const Eigen::VectorXd&);

  // Fit the linear regression model
  void fit();

  // Predict target values for new data
  Eigen::VectorXd predict(const Eigen::MatrixXd&) const;

  // Get the distance to prediction for each input row
  Eigen::VectorXd distanceToModel(const Eigen::MatrixXd&, const Eigen::VectorXd&) const;

  // Get the coefficients of the model
  Eigen::VectorXd getCoefficients() const;

  // Compute the avg of squared residuals
  double avgSquaredResiduals() const;
  
  // Compute the R^2 score of the model
  double score() const;
};

#endif /* LINEAR_REGRESSION_HPP */
