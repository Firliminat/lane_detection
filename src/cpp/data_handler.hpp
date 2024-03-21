#ifndef DATA_HANDLER_HPP
#define DATA_HANDLER_HPP

#include <vector>
#include <filesystem>
#include <Eigen/Dense>

class DataHandler {
protected:
  // Unique instance of the singleton
  static DataHandler* data_handler_;

  // Private constructor to prevent instantiation
  DataHandler(
    const std::string& data_folder,
    const std::string& lanes_folder,
    const int num_point_attributes
  );

public:
  // Folder where we can find files describing the points cloud
  std::filesystem::path data_folder;
  // Folder where we can find files describing the lanes
  std::filesystem::path lanes_folder;
  // Paths to the files containing the lidar points clouds
  std::vector<std::filesystem::path> lidar_paths;
  // Paths to the files containing the lanes coefficients
  std::vector<std::filesystem::path> lanes_paths;

  // Number of attributes for each point
  int num_point_attributes;
  // Number of frames in the data set
  int num_frames;

  // Get instance of the singleton
  static DataHandler *getInstance(const std::string&, const std::string&, const int);

  // Delete copy constructor and assignment operator to prevent copies
  DataHandler(const DataHandler&) = delete;
  DataHandler& operator=(const DataHandler&) = delete;

  // Reads from file the points corresponding to the given frame index
  void parseFolders();

  // Reads from file the points corresponding to the given frame index
  
  Eigen::MatrixXd readPoints(const int);

  // Reads from file the lanes corresponding to the given frame index
  Eigen::MatrixXd readLanes(const int);

  // Write lanes coefficients to the file corresponding to frame index
  void writeLanes(const int,const Eigen::MatrixXd&);
};

#endif /* DATA_HANDLER_HPP */