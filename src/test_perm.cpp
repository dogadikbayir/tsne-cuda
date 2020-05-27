#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
using namespace std;

// for string delimiter
vector<string> split (string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

int main() {
    
    std::vector<int> perm;
    
    std::string line;
    fstream myfile ("./perm.txt");
    if (myfile.is_open())
    {
      getline(myfile, line);
      std::vector<string> perms = split(line, " ");
      for (auto i : perms) perm.push_back(atoi(i.c_str()));
      
      std::cout << std::endl << "Read the perm vector: " << std::endl;
      for (auto p : perm) std::cout << p << " ";

      std::cout << std::endl;

    }
    return 0;
}
