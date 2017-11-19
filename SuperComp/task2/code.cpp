//
// Created by Ayat ".ospanoff" Ospanov
//

#include <cstdlib>
#include <iostream>

#include "ConjugateGradientMethod.h"
#include "DirichletProblemParams.h"

int main(int argc, char** argv) {
    try {
        MPIHelper MPIhelper;
        MPIhelper.Init(&argc, &argv);

        if (argc != 3) {
            throw CGMException("Usage: dirichlet <grid size by X> <grid size by Y>");
        }

        int gridSizeX = static_cast<int>(std::strtol(argv[1], NULL, 10));
        int gridSizeY = static_cast<int>(std::strtol(argv[2], NULL, 10));

        DirichletProblemParams dirichlet;
        ConjugateGradientMethod solver(dirichlet, gridSizeX, gridSizeY, MPIhelper);
        solver.solve();
        solver.collectResults();

        MPIhelper.Finalize();
    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        MPIHelper::Abort(1);
        return 1;
    } catch (...) {
        std::cerr << "Unknown Error!!!" << std::endl;
        MPIHelper::Abort(2);
        return 2;
    }
}
