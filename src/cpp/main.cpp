#include <vector>
#include "data_handler.hpp"

int main(){
  DataHandler* data_handler = DataHandler::GetInstance("..\\pointclouds", "..\\sample_output", 5);
  data_handler->ParseFolders();

  std::vector<std::vector<double>> points = data_handler->ReadPoints(0);
  std::vector<std::vector<double>> lanes_coefs = data_handler->ReadLanes(0);

  data_handler->WriteLanesCoefs(0, lanes_coefs);

  return 0;
}