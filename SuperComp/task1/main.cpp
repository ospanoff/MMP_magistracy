#include <omp.h>
#include <iostream>

int main(int argc, char const *argv[]) {
    std::cout << omp_get_max_threads();
    return 0;
}
