#include <iostream>
#include <Eigen\Dense>

#include "tools.hpp"
#include "data_handler.hpp"
#include "lanes_detector.hpp"

// Fits the scaling parameters to the first seven frames
// Then test for generalisation on the 2 remaining frames
void scalingLineDetectionParameters() {
  int num_features = 5;
  DataHandler* data_handler = DataHandler::getInstance(".\\pointclouds", ".\\sample_output", num_features);

  int num_frames_fitting = 7;

  int degree = 3;

  // Intensity scaling
  float max_min_intensity = 20.0;
  float min_min_intensity = 1.0;
  float min_intensity_step = 1.0;
  float max_max_intensity = 60.0;

  // Lane width scaling
  float max_max_lane_width = 5.0;
  float min_max_lane_width = 0.05;
  float min_min_lane_width = 0.05;
  float lane_width_step = 0.05;

  // Ridge regression params scaling
  float max_lambda_road_profile = 1.0;
  float min_lambda_road_profile = 0.001;
  float max_lambda_lanes = 1.0;
  float min_lambda_lanes = 0.001;
  float lambda_scaling = std::sqrt(10);

  std::cout << "Fitting the parameters" << std::endl;

  Tools::printTitles({
    "min_intensity",
    "min_intensity",
    "max_lane_width",
    "min_lane_width",
    "lambda_road_profile",
    "lambda_lanes",
    "best_score",
    "new_score"
  }, 21);

  float best_score = -1.0;
  float best_min_intensity = min_min_intensity;
  float best_max_intensity = min_min_intensity;
  float best_max_lane_width = min_max_lane_width;
  float best_min_lane_width = min_min_lane_width;
  float best_lambda_road_profile = max_lambda_road_profile;
  float best_lambda_lanes = max_lambda_lanes;
  for (float min_intensity = min_min_intensity; min_intensity <= max_min_intensity; min_intensity += min_intensity_step) {
    
    float min_max_intensity = min_intensity;
    float max_intensity_step = (max_max_intensity - min_intensity) / 5;
    for(float max_intensity = min_max_intensity; max_intensity <= max_max_intensity; max_intensity -= max_intensity_step) {

      for(float max_lane_width = min_max_lane_width; max_lane_width <= max_max_lane_width; max_lane_width += lane_width_step) {

        float max_min_lane_width = max_lane_width;
        for(float min_lane_width = min_min_lane_width; min_lane_width <= max_min_lane_width; min_lane_width += lane_width_step) {

          float lane_width_delta = (max_lane_width - min_lane_width) / 20.0;
          for (float lambda_road_profile = max_lambda_road_profile; lambda_road_profile >= min_lambda_road_profile; lambda_road_profile /= lambda_scaling){

            for (float lambda_lanes = max_lambda_lanes; lambda_lanes >= min_lambda_lanes; lambda_lanes /= lambda_scaling) {

              float new_score = 0;
              for (int frame_idx = 0; frame_idx < num_frames_fitting; ++frame_idx) {
                
                Eigen::MatrixXf points = data_handler->readPoints(frame_idx);

                LanesDetector lanes_detector(
                  points,
                  degree,
                  max_lane_width,
                  min_lane_width,
                  lane_width_delta,
                  max_intensity,
                  min_intensity,
                  lambda_road_profile,
                  lambda_lanes
                );
                lanes_detector.fit();
                
                new_score += lanes_detector.score() / num_frames_fitting;
              }
              
              Tools::printRow({
                best_min_intensity,
                best_min_intensity,
                best_max_lane_width,
                best_min_lane_width,
                best_lambda_road_profile,
                best_lambda_lanes,
                best_score,
                new_score
              }, 21);

              if (new_score > best_score) {
                best_score = new_score;
                best_min_intensity = min_intensity;
                best_max_intensity = min_intensity;
                best_max_lane_width = max_lane_width;
                best_min_lane_width = min_lane_width;
                best_lambda_road_profile = lambda_road_profile;
                best_lambda_lanes = lambda_lanes;
              }
            }
          }
        }
      }
    }
  }

  std::cout << "Fitting score: " << best_score << std::endl;
  std::cout << "Best parameters:" << std::endl;
  Tools::printTitles({
    "min_intensity",
    "min_intensity",
    "max_lane_width",
    "min_lane_width",
    "lambda_road_profile",
    "lambda_lanes"
  }, 21);
  Tools::printRow({
    best_min_intensity,
    best_min_intensity,
    best_max_lane_width,
    best_min_lane_width,
    best_lambda_road_profile,
    best_lambda_lanes
  }, 21);

  std::cout << "Generalising the parameters" << std::endl;

  float lane_width_delta = (best_max_lane_width - best_min_lane_width) / 20.0;
  float gen_score = 0.0;
  for (int frame_idx = num_frames_fitting; frame_idx < data_handler->num_frames; ++frame_idx) {
    Eigen::MatrixXf points = data_handler->readPoints(frame_idx);

    LanesDetector lanes_detector(
      points,
      degree,
      best_max_lane_width,
      best_min_lane_width,
      lane_width_delta,
      best_max_intensity,
      best_min_intensity,
      best_lambda_road_profile,
      best_lambda_lanes
    );
    lanes_detector.fit(true);

    data_handler->writeLanes(frame_idx, lanes_detector.getLanesCoefficients());
    
    gen_score += lanes_detector.score() / (data_handler->num_frames - num_frames_fitting);
  }
  
  std::cout << "Generalisation score: " << gen_score << std::endl;
}

// Applies the regression to all of the frames and outputs the lanes coefficients in the corresponding folders
int main() {

  const std::string data_folder = ".\\pointclouds";
  const std::string lanes_folder = ".\\sample_output";
  int num_features = 5;

  DataHandler* data_handler = DataHandler::getInstance(".\\pointclouds", ".\\sample_output", num_features);

  // Model Parameters
  int degree = 3;

  float max_lane_width = 3.0;
  float min_lane_width = 1.8;
  float lane_width_delta = 0.05;

  float max_intensity = 20.0;
  float min_intensity = 3.0;

  float lambda_road_profile = 1.0;
  float lambda_lanes = 0.0;

  for (int frame_idx = 0; frame_idx < data_handler->num_frames; ++frame_idx) {

    std::cout << "Frame: " << frame_idx << std::endl;
    std::cout << "Loading data" << std::endl;
    Eigen::MatrixXf points = data_handler->readPoints(frame_idx);

    std::cout << "Fitting the model to the data" << std::endl;
    LanesDetector lanes_detector(
      points,
      degree,
      max_lane_width,
      min_lane_width,
      lane_width_delta,
      max_intensity,
      min_intensity,
      lambda_road_profile,
      lambda_lanes
    );
    lanes_detector.fit(true);

    data_handler->writeLanes(frame_idx, lanes_detector.getLanesCoefficients());
  }

  return 0;
}