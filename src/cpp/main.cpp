#include <vector>
#include <iostream>
#include <Eigen\Dense>
#include "data_handler.hpp"

int main(){

  DataHandler* data_handler = DataHandler::getInstance("..\\..\\pointclouds", "..\\..\\sample_output", 5);
  data_handler->parseFolders();

  Eigen::MatrixXd points = data_handler->readPoints(0);
  Eigen::MatrixXd lanes = data_handler->readLanes(0);
  std::cout << "lanes:\n" << lanes << std::endl;

  Eigen::MatrixXd bias = Eigen::MatrixXd::Ones(points.rows(), 1);
  Eigen::MatrixXd x(points.col(0));
  Eigen::MatrixXd x_squared(x.array() * x.array());
  Eigen::MatrixXd X(x.rows(), bias.cols()+x.cols()+x_squared.cols());
  X << bias, x, x_squared;
  Eigen::VectorXd y = Eigen::MatrixXd(points).col(1);
  std::cout << "The solution using the QR decomposition is:\n"
      << X.colPivHouseholderQr().solve(y) << std::endl;
}