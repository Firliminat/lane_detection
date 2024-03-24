#include "multi_polynomial_regression.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>

#include "polynomial_regression.hpp"
#include "tools.hpp"


// Counts the number of points assigned to each model
Eigen::VectorXi MultiPolynomialRegression::countAssignedPoints() const {
  Eigen::VectorXi assigned_counts = Eigen::VectorXi::Zero(poly_models.size());
  for (int sample_idx = 0; sample_idx < X.rows(); ++sample_idx){
    assigned_counts(a(sample_idx)) += 1;
  }

  return assigned_counts;
}

// Updates the given inputs, targets and weights with the ones assigned to model with given index
void MultiPolynomialRegression::updateWithAssignedData(
  Eigen::MatrixXf& assigned_X,
  Eigen::VectorXf& assigned_y,
  Eigen::VectorXf& assigned_w,
  const int model_idx
) const {
  assigned_X = Eigen::MatrixXf(0, X.cols());
  assigned_y = Eigen::VectorXf(0);
  assigned_w = Eigen::VectorXf(0);
  for (int sample_idx = 0; sample_idx < X.rows(); ++sample_idx){
    if(a(sample_idx) == model_idx) {
      assigned_X.conservativeResize(assigned_X.rows() + 1, Eigen::NoChange);
      assigned_X.row(assigned_X.rows() - 1) = X.row(sample_idx);

      assigned_y.conservativeResize(assigned_y.rows() +1 );
      assigned_y(assigned_y.rows() - 1) = y(sample_idx);

      assigned_w.conservativeResize(assigned_w.rows() + 1);
      assigned_w(assigned_w.rows() - 1) = w(sample_idx);
    }
  }
}

// Assigns the data points tothe closest polynomial
void MultiPolynomialRegression::assignToModels(Eigen::VectorXi& assignments) const {
  // Making sure the assignment vector has the right size
  if (assignments.size() != X.rows()) {
    assignments.conservativeResize(X.rows());
  }

  Eigen::MatrixXf squaredDistances = squaredDistancesToModels(X, y);
  // For each point iterates over the distances to models to assign the closest model
  for (int sample_idx = 0; sample_idx < X.rows(); ++sample_idx) {
    float min_dist = std::numeric_limits<float>::max();

    for (int model_idx = 0; model_idx < poly_models.size(); ++model_idx) {
      float new_dist = squaredDistances(sample_idx, model_idx);

      if (new_dist < min_dist) {
        min_dist = new_dist;
        assignments(sample_idx) = model_idx;
      }
    }
  }
}

// Initialize the polynomial models
void MultiPolynomialRegression::initModels(const int num_models){
  if (num_models < 1) {
    throw std::invalid_argument("Invalidnumber of polynomial models.");
  }
  poly_models = Eigen::VectorX<PolynomialRegression>(num_models);
  float min_y = y.minCoeff(), delta_y = y.maxCoeff() - min_y;
  for (int model_idx = 0; model_idx < num_models; ++model_idx) {
    poly_models(model_idx) = PolynomialRegression(lambda, degree);
    Eigen::VectorXf coefficients(degree + 1);
    
    coefficients.block(0, 0, degree, 1) = Eigen::VectorXf::Zero(degree);
    coefficients(degree) = (model_idx + 1) * delta_y / (num_models + 1) + min_y;

    poly_models(model_idx).setCoefficients(coefficients);
  }
}

// fit the polynomial models to the data points assigned to them
void MultiPolynomialRegression::fitModels(){
  for (int model_idx = 0; model_idx < poly_models.size(); ++model_idx) {
    Eigen::MatrixXf assigned_X;
    Eigen::VectorXf assigned_y;
    Eigen::VectorXf assigned_w;
    updateWithAssignedData(assigned_X, assigned_y, assigned_w, model_idx);

    poly_models(model_idx).updateData(assigned_X, assigned_y, assigned_w);
    poly_models(model_idx).fit();
  }
}


MultiPolynomialRegression::MultiPolynomialRegression(
  const Eigen::MatrixXf& inputs,
  const Eigen::VectorXf& targets,
  const Eigen::VectorXf& weights,
  const float lambda,
  const int degree,
  const int num_models
) {
  X = Eigen::MatrixXf(inputs);
  y = Eigen::VectorXf(targets);
  a = Eigen::VectorXi(X.rows());
  this->degree = degree;
  this->lambda = lambda;

  // If weights are provided, use them; otherwise, initialize weights to ones
  if (weights.size() < 1) {
    w = Eigen::VectorXf::Ones(inputs.rows());
  } else {
    if (weights.size() != inputs.rows()) {
      throw std::invalid_argument("Invalid size of weights vector.");
    }
    w = Eigen::VectorXf(weights);
  }
  // Making sure weights are positive
  w = Eigen::VectorXf(w.array().abs());
  // and weights vector is unitary for norm 1
  w = Eigen::VectorXf(w.array() / w.sum());

  // Initialize the models
  initModels(std::max(num_models, 1));
}

// Fit the model
void MultiPolynomialRegression::fit(bool verbose) {
  if(verbose) {
    std::vector<std::string> titles = {"iter", "score", "improov"};
    for (int model_idx = 0; model_idx < poly_models.size(); ++model_idx) {
      titles.push_back("score " + std::to_string(model_idx));
      titles.push_back("count " + std::to_string(model_idx));
    }
    Tools::printTitles(titles, 13);
  }

  bool score_condition = true;
  float
    old_score,
    new_score = std::numeric_limits<float>::max(),
    min_improvement = 0.5;
  int iter_idx = 0;
  while (score_condition && iter_idx++ < 50) {
    old_score = new_score;

    assignToModels(a);
    fitModels();
    
    new_score = weightedSumSquaredResiduals();
    score_condition = old_score - new_score > min_improvement;

    if(verbose) {
      Eigen::MatrixXi assigned_counts = countAssignedPoints();
      
      std::vector<float> data = {static_cast<float>(iter_idx), new_score, old_score - new_score};
      for (int model_idx = 0; model_idx < poly_models.size(); ++model_idx) {
        data.push_back(poly_models(model_idx).weightedSumSquaredResiduals());
        data.push_back(assigned_counts(model_idx));
      }
      Tools::printRow(data, 13);
    }
  }
}

// Fit the number of models
void MultiPolynomialRegression::fitNumModels(
  const int min_num_models,
  const int max_num_models,
  bool verbose,
  bool talkative
) {
  if(verbose && !talkative) {
    Tools::printTitles({"num models", "silhouette", "improov", "sum resi", "avg R^2"}, 13);
  }

  bool score_condition = true;
  float
    old_score,
    new_score = std::numeric_limits<float>::max(),
    min_improvement = 0.05;
  int num_models = std::max(min_num_models, 1) - 1;
  while (score_condition && num_models++ < max_num_models) {
    old_score = new_score;

    // Initialize the models vector with the wished number of models
    initModels(num_models);
    // fit the models
    fit(talkative);
    
    // Update the looping conditions
    new_score = silhouetteScore();
    score_condition = old_score - new_score > min_improvement;
    if(verbose) {
      if(talkative) {
        Tools::printTitles({"num models", "silhouette", "improov", "sum resi", "avg R^2"}, 13);
      }
      Tools::printRow({static_cast<float>(num_models), new_score, old_score - new_score, weightedSumSquaredResiduals(), avgScore()}, 13);
    }
  }

  // If the last iteration decreased the score
  // We go back to the previous one
  if (old_score - new_score < 0) {
    --num_models;

    initModels(num_models);
    fit(talkative);
    if(verbose) {
      // Update the looping conditions
      new_score = silhouetteScore();
      score_condition = old_score - new_score > min_improvement;
      if(talkative) {
        Tools::printTitles({"num models", "silhouette", "improov", "sum resi", "avg R^2"}, 13);
      }
      Tools::printRow({static_cast<float>(num_models), new_score, old_score - new_score, weightedSumSquaredResiduals(), avgScore()}, 13);
    }
  }
}

// Predict target values for new data for all models
Eigen::MatrixXf MultiPolynomialRegression::predict(const Eigen::MatrixXf& new_data) const {
  int num_models = poly_models.size();
  Eigen::MatrixXf y_predicted(new_data.rows(), num_models);
  for (int model_idx = 0; model_idx < num_models; ++model_idx) {
    y_predicted.col(model_idx) = poly_models(model_idx).predict(new_data);
  }

  return y_predicted;
}

// Get the squared distances to prediction for each model
Eigen::MatrixXf MultiPolynomialRegression::squaredDistancesToModels(
  const Eigen::MatrixXf& inputs,
  const Eigen::VectorXf& targets
) const {
  int num_models = poly_models.size();
  Eigen::MatrixXf squaredDistances(inputs.rows(), num_models);
  for (int modelidx = 0; modelidx < num_models; ++modelidx) {
    PolynomialRegression model = poly_models(modelidx);
    squaredDistances.col(modelidx) = model.squaredDistancesToModel(inputs, targets);
  }
  return squaredDistances;
}

// Get the squared distance to assigned model for each row
Eigen::VectorXf MultiPolynomialRegression::squaredDistancesToModel(
  const Eigen::MatrixXf& inputs,
  const Eigen::VectorXf& targets,
  const Eigen::VectorXi& assignments
) const {
  // If we have no assignment we use this->a
  Eigen::MatrixXi safe_assignments(assignments);
  if (safe_assignments.size() < inputs.rows()) {
    safe_assignments = a;
  }

  int num_points = inputs.rows();
  Eigen::VectorXf squaredDistances(num_points);
  for (int sample_idx = 0; sample_idx < num_points; ++sample_idx) {
    PolynomialRegression model = poly_models(safe_assignments(sample_idx)); 
    squaredDistances.row(sample_idx) = model.squaredDistancesToModel(
      inputs.row(sample_idx),
      targets.row(sample_idx)
    );
  }

  return squaredDistances;
}

// Get the coefficients of the models
Eigen::MatrixXf MultiPolynomialRegression::getCoefficients() const {
  int num_models = poly_models.size();
  Eigen::MatrixXf coefficients(num_models, degree + 1);
  for (int model_idx = 0; model_idx < num_models; ++model_idx) {
    coefficients.row(model_idx) = poly_models(model_idx).getCoefficients();
  }
  return coefficients;
}

// Compute the the sum of squared residuals to assigned model
float MultiPolynomialRegression::weightedSumSquaredResiduals() const {
  Eigen::VectorXf squaredDistances = squaredDistancesToModel(X, y, a);
  return squaredDistances.sum();
}

// Compute the average model wise of the sum squared residuals
float MultiPolynomialRegression::avgSquaredResiduals() const {
  int num_models = poly_models.size();
  float avgSquaredResiduals = 0.0;
  for (int model_idx = 0; model_idx < num_models; ++model_idx) {
    avgSquaredResiduals += poly_models(model_idx).weightedSumSquaredResiduals();
  }
  return avgSquaredResiduals / num_models;
}

// Compute the average model wise of the R^2 score
float MultiPolynomialRegression::avgScore() const {
  int num_models = poly_models.size();
  float score = 0.0;
  for (int model_idx = 0; model_idx < num_models; ++model_idx) {
    score += poly_models(model_idx).score();
  }
  return score / num_models;
}
  
// Computes the simplified silhouette score of the model
float MultiPolynomialRegression::silhouetteScore() const {
  Eigen::VectorXf silhouette_scores = Eigen::VectorXf::Zero(poly_models.size());
  Eigen::MatrixXf squared_distances = squaredDistancesToModels(X, y);

  // For each point iterates over the distances to models to assign the closest model
  for (int sample_idx = 0; sample_idx < X.rows(); ++sample_idx) {
    float 
      min_dist = std::numeric_limits<float>::max(),
      second_min_dist = std::numeric_limits<float>::max();

    for (int model_idx = 0; model_idx < poly_models.size(); ++model_idx) {
      float new_dist = squared_distances(sample_idx, model_idx);

      if (new_dist < min_dist) {
        second_min_dist = min_dist;
        min_dist = new_dist;
      }
      else if (new_dist < second_min_dist)
      {
        second_min_dist = new_dist;
      }
    }
    float point_silhouette_score = w(sample_idx) * (1 - min_dist / second_min_dist);
    if (std::isnan(point_silhouette_score)) {
      silhouette_scores(a(sample_idx)) += point_silhouette_score;
    }
  }

  return silhouette_scores.sum() / X.rows();
}
