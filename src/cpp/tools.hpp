#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm> 
#include <cctype>
#include <locale>
#include <Eigen\Dense>

#include "polynomial_regression.hpp"


namespace Tools {
  // trim whitespaces from start (in place)
  inline void ltrim(std::string &s) {
      s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
          return !std::isspace(ch);
      }));
  }

  // trim whitespaces from end (in place)
  inline void rtrim(std::string &s) {
      s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
          return !std::isspace(ch);
      }).base(), s.end());
  }

  // trim whitespaces from both ends (in place)
  inline void trim(std::string &s) {
    rtrim(s);
    ltrim(s);
  }

  std::vector<std::string> split(
    std::string str, 
    std::string delimiter
  );

  // Prints the titles
  void printTitles(const std::vector<std::string>&, const std::streamsize = 15);

  // Prints a row
  void printRow(
    const std::vector<float>&,
    const std::streamsize = 15
  );

  Eigen::VectorXf clip(
    const Eigen::VectorXf&,
    const float = 0.0,
    const float = 255.
  );

  // Row-wise filtering based on given colomn and interval
  // We keep only inputs such that min <= input.col(dim) <= max
  Eigen::MatrixXf rowWiseFilter(
    const Eigen::MatrixXf& inputs = Eigen::MatrixXf::Zero(0,0),
    const Eigen::Index dim = 0,
    const float min = std::numeric_limits<float>::min(),
    const float max = std::numeric_limits<float>::max()
  );

}

#endif /* TOOLS_HPP */