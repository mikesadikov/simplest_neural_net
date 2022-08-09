
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int readCsv(std::string csv_file_train, std::string csv_file_test,
            std::vector<float>& x, std::vector<float>& y, std::vector<float>& t,
            std::vector<int>& t_y);
