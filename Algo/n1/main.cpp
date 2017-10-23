#include <iostream>
#include <fstream>

#include "closest_points.h"

int main(int argc, char const *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: ./closest_pts <input file>" << std::endl;
    }
    std::string fname = argv[1];
    std::ifstream file(fname.c_str());

    if (file.is_open()) {
        std::uint32_t N;
        file >> N;
        std::vector<Point> pts(N);

        for (std::uint32_t i = 0; i < N; ++i) {
            file >> pts[i].x >> pts[i].y;
            pts[i].id = i;
        }
        file.close();

        auto finder = ClosestPointsFinder(pts);
        finder.find_2closest_points();
        std::cout << "Файл: " << fname << std::endl;
        std::cout << finder.get_fancy_results();

    } else {
        std::cerr << "File " << fname << " can't be opened!" << std::endl;
    }

    return 0;
}
