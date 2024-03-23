#include "multi_polynomial_regression.hpp"

#include <Eigen/Dense>
#include "polynomial_regression.hpp"

// Updates the given inputs, targets and weights with the ones assigned to model with given index
void MultiPolynomialRegression::updateWithAssignedData(
  Eigen::MatrixXd& assigned_X,
  Eigen::VectorXd& assigned_y,
  Eigen::VectorXd& assigned_weights,
  int model_idx
) const {
  assigned_X = Eigen::MatrixXd(0, X.cols());
  assigned_y = Eigen::VectorXd(0);
  assigned_weights = Eigen::VectorXd(0);
  for (int point_idx = 0; point_idx < X.rows(); ++point_idx){
    if(points_assignments(point_idx) == model_idx) {
      assigned_X.conservativeResize(assigned_X.rows() + 1, Eigen::NoChange);
      assigned_X.row(assigned_X.rows() - 1) = X.row(point_idx);

      assigned_y.conservativeResize(assigned_y.rows() +1 );
      assigned_y(assigned_y.rows() - 1) = y(point_idx);

      assigned_weights.conservativeResize(assigned_weights.rows() + 1);
      assigned_weights(assigned_weights.rows() - 1) = weights(point_idx);
    }
  }
}

// Assigns the data points tothe closest polynomial
void MultiPolynomialRegression::assignToPolynomials(){
  // For each point iterates over the models to assign the closest one
  for (int input_idx = 0; input_idx < X.rows(); ++input_idx) {
    double min_dist = std::numeric_limits<double>::max();
    for (int model_idx = 0; model_idx < poly_models.size(); ++model_idx) {
      Eigen::MatrixXd inputs = X.row(input_idx);
      Eigen::VectorXd targets = y.row(input_idx);
      double new_dist = poly_models(model_idx).distanceToModel(inputs, targets)(0);
      if (new_dist < min_dist) {
        min_dist = new_dist;
        points_assignments(input_idx) = model_idx;
      }
    }
  }
}

// Initialize the polynomial models
void MultiPolynomialRegression::initModels(int num_models){
  if (num_models < 1) {
    throw std::invalid_argument("Invalidnumber of polynomial models.");
  }
  poly_models = Eigen::VectorX<PolynomialRegression>(num_models);
  double min_y = y.minCoeff(), delta_y = y.maxCoeff() - min_y;
  for (int model_idx = 0; model_idx < num_models; ++model_idx) {
    poly_models(model_idx) = PolynomialRegression(lambda, degree);
    Eigen::VectorXd coefficients(degree + 1);
    coefficients.block(0, 0, degree, 1) = Eigen::VectorXd::Zero(degree);
    coefficients(degree) = (model_idx + 1) * delta_y / (num_models + 1) + min_y;
    poly_models(model_idx).setCoefficients(coefficients);
  }
}

// fit the polynomial models to the data points assigned to them
void MultiPolynomialRegression::fitModels(){
  for (int model_idx = 0; model_idx < poly_models.size(); ++model_idx) {
    Eigen::MatrixXd assigned_X;
    Eigen::VectorXd assigned_y;
    Eigen::VectorXd assigned_weights;
    updateWithAssignedData(assigned_X, assigned_y, assigned_weights, model_idx);

    poly_models(model_idx).updateData(assigned_X, assigned_y, assigned_weights);
    poly_models(model_idx).fit();
  }
}


MultiPolynomialRegression::MultiPolynomialRegression(
  const Eigen::MatrixXd& inputs,
  const Eigen::VectorXd& targets,
  const Eigen::VectorXd& weights,
  double lambda,
  int degree,
  int num_models
) :
  X(inputs),
  weights(weights),
  points_assignments(X.rows())
{
  this->y = Eigen::VectorXd(targets);
  this->degree = degree;
  this->lambda = lambda;
  initModels(num_models);
}

// Fit the model
void MultiPolynomialRegression::fit() {
  bool score_condition = true;
  double
    old_score,
    new_score = std::numeric_limits<double>::max(),
    min_improvement = 1000.0;
  int iter_idx = 0;
  while (score_condition && iter_idx++ < 50) {
    old_score = new_score;

    assignToPolynomials();
    fitModels();
    
    new_score = avgSquaredResiduals();
    score_condition = old_score - new_score > min_improvement;
  }
}

// Predict target values for new data for all models
Eigen::MatrixXd MultiPolynomialRegression::predict(const Eigen::MatrixXd& new_data) const {
  int num_models = poly_models.size();
  Eigen::MatrixXd y_predicted(new_data.rows(), num_models);
  for (int model_idx = 0; model_idx < num_models; ++model_idx) {
    y_predicted.col(model_idx) = poly_models(model_idx).predict(new_data);
  }

  return y_predicted;
}

// Get the coefficients of the models
Eigen::MatrixXd MultiPolynomialRegression::getCoefficients() const {
  int num_models = poly_models.size();
  Eigen::MatrixXd coefficients(num_models, degree + 1);
  for (int model_idx = 0; model_idx < num_models; ++model_idx) {
    coefficients.row(model_idx) = poly_models(model_idx).getCoefficients();
  }
  return coefficients;
}

// Compute the avg of squared residuals
double MultiPolynomialRegression::avgSquaredResiduals() const {
  int num_models = poly_models.size();
  double avgSquaredResiduals = 0.0;
  for (int model_idx = 0; model_idx < num_models; ++model_idx) {
    avgSquaredResiduals += poly_models(model_idx).avgSquaredResiduals();
  }
  return avgSquaredResiduals / num_models;
}

// Compute the average R^2 score of the models
double MultiPolynomialRegression::avgScore() const {
  int num_models = poly_models.size();
  double score = 0.0;
  for (int model_idx = 0; model_idx < num_models; ++model_idx) {
    score += poly_models(model_idx).score();
  }
  return score / num_models;
}
