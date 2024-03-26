#include "tools.hpp"

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
  /* separate the string in a vector of string using the
  given delimiter */
  std::vector<std::string> split(
    std::string str, 
    std::string delimiter
  ) {
    std::string s(str); 
    size_t pos = 0;
    std::vector<std::string> substrings{};
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
      token = s.substr(0, pos);
      substrings.push_back(token);
      s.erase(0, pos + delimiter.length());
    }
    
    substrings.push_back(s);
    return substrings;
  }

  // Prints the titles
  void printTitles(
    const std::vector<std::string>& titles,
    const std::streamsize width
  ) {
    // Print column titles
    std::cout << std::left;
    for (const auto& title : titles) {
      std::cout << std::setw(width) << title;
      std::cout << " | ";
    }
    std::cout << std::endl;
  }

  // Prints a row
  void printRow(
    const std::vector<float>& row,
    const std::streamsize width
  ) {
    for (const auto& value : row) {
      std::cout << std::setw(width) << value;
      std::cout << " | ";
    }
    std::cout << std::endl;
  }

  // Clips the value of the vector to the given interval
  Eigen::VectorXf clip(
    const Eigen::VectorXf& inputs,
    const float min,
    const float max
  ) {
    Eigen::VectorXf outputs(inputs.size());
    for (int input_idx = 0; input_idx < inputs.size(); ++input_idx) {
      if (min > inputs(input_idx)) {
        outputs(input_idx) = min;
      } else if (inputs(input_idx) > max) {
        outputs(input_idx) = max;
      }
    }
    return outputs;
  }

  // Row-wise filtering based on given colomn and interval
  // We keep only inputs such that min < input.col(col_idx) < max
  Eigen::MatrixXf rowWiseFilter(
    const Eigen::MatrixXf& inputs,
    const Eigen::Index col_idx,
    const float min,
    const float max
  ) {
    Eigen::MatrixXf filtered_inputs(0, inputs.cols());
    for (int pointidx = 0; pointidx < inputs.rows(); ++pointidx) {
      if (min < inputs(pointidx, col_idx) && inputs(pointidx, col_idx) < max) {
        filtered_inputs.conservativeResize(filtered_inputs.rows() + 1, Eigen::NoChange);
        filtered_inputs.row(filtered_inputs.rows() - 1) = inputs.row(pointidx);
      }
    }
    return filtered_inputs;
  }
}
