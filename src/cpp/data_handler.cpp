#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include "data_handler.hpp"
#include "tools.hpp"


// Unique instance of the singleton
DataHandler* DataHandler::data_handler_ = nullptr;

// Private constructor to prevent instantiation
DataHandler::DataHandler(
  const std::string& data_folder,
  const std::string& lanes_folder,
  const int num_point_attributes
) :
  data_folder(data_folder),
  lanes_folder(lanes_folder),
  num_point_attributes(num_point_attributes)
{}

// Get instance of the singleton
DataHandler *DataHandler::GetInstance(
  const std::string& data_folder,
  const std::string& lanes_folder,
  const int num_point_attributes
)
{
    if(data_handler_==nullptr){
        data_handler_ = new DataHandler(data_folder, lanes_folder, num_point_attributes);
    }
    return data_handler_;
}

// Reads from file the points corresponding to the given frame index
void DataHandler::ParseFolders() {
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

  std::sort(this->lidar_paths.begin(), this->lidar_paths.end());
  std::sort(this->lanes_paths.begin(), this->lanes_paths.end());
}

// Reads from file the points corresponding to the given frame index
std::vector<std::vector<double>> DataHandler::ReadPoints(
  // Index of the frame to read
  const int frame_index
) {
  std::ifstream lidar_file(
    this->lidar_paths.at(frame_index),
    std::ios::binary
  );
  if (!lidar_file.is_open()) {
    throw std::runtime_error("Failed to open file for reading.");
  }

  std::vector<std::vector<double>> points{};
  std::vector<double> point(this->num_point_attributes, 0.0);
  double value = 0.0;
  int attribute_index = 0;
  while(lidar_file.read(reinterpret_cast<char*>(&value), sizeof(double))){
    point.at(attribute_index) = value;

    if(attribute_index != (this->num_point_attributes -1)) {
      ++attribute_index;
    }
    else {
      points.push_back(point);
      attribute_index = 0;
    }
  }

  lidar_file.close();
  return points;
}

// Reads from file the lanes corresponding to the given frame index
std::vector<std::vector<double>> DataHandler::ReadLanes(
  // Index of the frame to read
  const int frame_index
) {
  std::ifstream lanes_file(this->lanes_paths.at(frame_index));
  if (!lanes_file.is_open()) {
    throw std::runtime_error("Failed to open file for reading.");
  }

  std::vector<std::vector<double>> lanes_coefs{};
  std::string line;
  while (std::getline(lanes_file, line)) {
    Tools::trim(line);
    std::vector<double> lane_coefs{};
    for (std::string coef_str : Tools::split(line, ";")) {
      double coef = stod(coef_str);
      lane_coefs.push_back(coef);
    }
    lanes_coefs.push_back(lane_coefs);
  }

  lanes_file.close();
  return lanes_coefs;
}

// Write lanes coefficients to the file corresponding to frame index
void DataHandler::WriteLanesCoefs(
  const int frame_index,
  const std::vector<std::vector<double>>& lanes_coefs
) {
  if(lanes_coefs.empty()) {
    return;
  }

  std::ofstream lane_file(this->lanes_paths.at(frame_index));
  if (!lane_file.is_open()) {
      throw std::runtime_error("Failed to open file for writing.");
  }

  std::stringstream file_stream;
  for (const auto& lane_coefs : lanes_coefs) {
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