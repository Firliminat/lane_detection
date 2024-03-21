#include "tools.hpp"

#include <string>
#include <vector>
#include <algorithm> 
#include <cctype>
#include <locale>
#include <Eigen\Dense>

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

  // Remove a row from Eigen Matrix
  void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
  {
      unsigned int numRows = matrix.rows()-1;
      unsigned int numCols = matrix.cols();

      if( rowToRemove < numRows )
          matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

      matrix.conservativeResize(numRows,numCols);
  }

  // Remove a column from Eigen Matrix
  void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove)
  {
      unsigned int numRows = matrix.rows();
      unsigned int numCols = matrix.cols()-1;

      if( colToRemove < numCols )
          matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

      matrix.conservativeResize(numRows,numCols);
  }
}