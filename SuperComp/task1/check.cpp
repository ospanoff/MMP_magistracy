#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>


int main(int argc, char *argv[]) {
    int size1, size2;

    double EPS = 0.001;
    if (argc < 3) {
        std::cout << "Usage: check <input file 1> <input file 2>" << std::endl;
        return 1;
    }
    if (argc == 4)
        EPS = atof(argv[3]);

    std::ifstream file1, file2;
    file1.open(argv[1]);
    file2.open(argv[2]);

    file1 >> size1;
    file2 >> size2;
    if (size1 != size2) {
        std::cout << "Wrong sizes!\n";
        return 1;
    }

    double tmp1, tmp2;
    for (int i = 0; i < size1; ++i) {
        file1 >> tmp1;
        file2 >> tmp2;
        if (std::fabs(tmp1 - tmp2) > EPS) {
            std::cout << "Wrong values: " << i << " " << tmp1 << " " << tmp2 << std::endl;
            return 1;
        }
    }
    file1.close();
    file2.close();

    return 0;
}
