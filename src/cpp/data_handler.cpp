#include "data_handler.hpp"

#include <fstream>
#include <filesystem>
#include <vector>
#include <Eigen/Dense>

#include "tools.hpp"


// Unique instance of the singleton
DataHandler* DataHandler::data_handler_ = nullptr;

// Private constructor to prevent instantiation
DataHandler::DataHandler(
  const std::string& data_folder,
  const std::string& lanes_folder,
  const int num_features
)
{
  this->update(
    data_folder,
    lanes_folder,
    num_features
  );
}

// Reads from file the points corresponding to the given frame index
void DataHandler::parseFolders() {
  this->lidar_paths = std::vector<std::filesystem::path>();
  this->lanes_paths = std::vector<std::filesystem::path>();
  this->num_frames = 0;
  for (const auto & entry : std::filesystem::directory_iterator(this->data_folder)) {
    std::filesystem::path lidar_path = entry.path();

    this->lidar_paths.push_back(lidar_path);

    std::filesystem::path lane_path = this->lanes_folder / lidar_path.replace_extension("txt").filename();
    this->lanes_paths.push_back(lane_path);

    this->num_frames += 1;
  }

  // Making sure the paths are in same order
  std::sort(this->lidar_paths.begin(), this->lidar_paths.end());
  std::sort(this->lanes_paths.begin(), this->lanes_paths.end());
}

// Get instance of the singleton
DataHandler *DataHandler::getInstance(
  const std::string& data_folder,
  const std::string& lanes_folder,
  const int num_features
)
{
    if(data_handler_ == nullptr){
      data_handler_ = new DataHandler(
        data_folder,
        lanes_folder,
        num_features
      );
    } else {
      data_handler_->update(
        data_folder,
        lanes_folder,
        num_features
      );
    }
    return data_handler_;
}

// update the properties and parse the folders
void DataHandler::update(
  const std::string& data_folder,
  const std::string& lanes_folder,
  const int num_features
) {
  this->data_folder = data_folder;
  this->lanes_folder = lanes_folder;
  this->num_features = num_features;

  this->parseFolders();
}

// Reads from file the points corresponding to the given frame index
Eigen::MatrixXf DataHandler::readPoints(
  const int frame_idx
) {
  std::ifstream lidar_file(
    this->lidar_paths.at(frame_idx),
    std::ios::binary
  );
  if (!lidar_file.is_open()) {
    throw std::runtime_error("Failed to open file for reading.");
  }

  // Determine the file length
  lidar_file.seekg(0, std::ios_base::end);
  std::size_t size = lidar_file.tellg();
  lidar_file.seekg(0, std::ios_base::beg);

  // Create a vector to store the data
  std::vector<float> input_buffer(size/sizeof(float));
  // Load the data
  lidar_file.read((char*) &input_buffer[0], size);

  // Map the data to a matrix 
  Eigen::MatrixXf points = Eigen::Map<
    Eigen::Matrix<
      float,
      Eigen::Dynamic,
      Eigen::Dynamic,
      Eigen::RowMajor
    >
  >(input_buffer.data(), input_buffer.size()/num_features, num_features);

  lidar_file.close();
  return points;
}

// Reads from file the lanes corresponding to the given frame index
Eigen::MatrixXf DataHandler::readLanes(
  // Index of the frame to read
  const int frame_idx
) {
  std::ifstream lanes_file(this->lanes_paths.at(frame_idx));
  if (!lanes_file.is_open()) {
    throw std::runtime_error("Failed to open file for reading.");
  }

  std::string line;
  std::vector<float> buff{};
  int num_rows = 0, num_cols;
  while (std::getline(lanes_file, line)) {
    Tools::trim(line);
    num_cols = 0;
    for (std::string coef_str : Tools::split(line, ";")) {
      float coef = stod(coef_str);
      buff.push_back(coef);
      ++num_cols;
    }
    ++num_rows;
  }

  lanes_file.close();
  
  Eigen::MatrixXf lanes_coefs = Eigen::MatrixXf::Zero(num_rows, num_cols);
  for(int i = 0; i < num_rows; ++i) {
    for(int j = 0; j < num_cols; ++j) {
      lanes_coefs(i, j) = buff.at(i * num_cols + j);
    }
  }

  return lanes_coefs;
}

// Write lanes coefficients to the file corresponding to frame index
void DataHandler::writeLanes(
  const int frame_idx,
  const Eigen::MatrixXf& lanes_coefs
) {
  if(lanes_coefs.rows() < 1) {
    return;
  }

  std::ofstream lane_file(this->lanes_paths.at(frame_idx));
  if (!lane_file.is_open()) {
    throw std::runtime_error("Failed to open file for writing.");
  }

  std::stringstream file_stream;
  for (const auto& lane_coefs : lanes_coefs.rowwise()) {
    std::string line_string;
    std::stringstream line_stream;
    for (const auto& coef : lane_coefs) {
      line_stream << std::scientific << std::setprecision(19);  
      line_stream << coef << ";";
    }
    line_string = line_stream.str();
    line_string.pop_back();
    file_stream << line_string << std::endl;
  }
  std::string file_string = file_stream.str();
  file_string.pop_back();
  lane_file << file_string;

  lane_file.close();
}