#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <string>
#include <vector>
#include <algorithm> 
#include <cctype>
#include <locale>
#include <Eigen\Dense>


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

  // Remove a row from Eigen Matrix
  void removeRow(Eigen::MatrixXd&, int);

  // Remove a column from Eigen Matrix
  void removeColumn(Eigen::MatrixXd&, int);
}

#endif /* TOOLS_HPP */