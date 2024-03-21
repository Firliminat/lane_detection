#include "tools.hpp"
#include <string>
#include <vector>
#include <algorithm> 
#include <cctype>
#include <locale>

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
}