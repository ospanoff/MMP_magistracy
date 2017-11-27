#include <iostream>
#include <cstdlib>

#include "MPIHelper.h"
#include "Grid.h"
#include "DirichletProblem.h"
#include "Func2D.h"
#include "ConjugateGradientMethod.h"


int main(int argc, char *argv[]) {
    try {
        MPIHelper &helper = MPIHelper::getInstance();
        helper.init(&argc, &argv);

        if (argc != 3) {
            throw std::string("Usage: dirichlet <grid size by X> <grid size by Y>");
        }

        unsigned int gridSizeX = static_cast<unsigned int>(std::strtol(argv[1], NULL, 10));
        unsigned int gridSizeY = static_cast<unsigned int>(std::strtol(argv[2], NULL, 10));

        Grid grid = Grid::getGrid(DirichletProblem::getBorders(), gridSizeX, gridSizeY);
        Func2D f(grid, DirichletProblem::F);

        double res = ~f * ~f;
        if (helper.isMaster()) {
            std::cout << res << std::endl;
        }

//        for (int i = 0; i < grid.x.size(); ++i) {
//            for (int j = 0; j < grid.y.size(); ++j) {
//                std::cout << f(i, j) << " ";
//            }
//            std::cout << std::endl;
//        }
//        ConjugateGradientMethod solver(grid, DirichletProblem::F, DirichletProblem::phi, false, DirichletProblem::answer);
//        solver.solve();
//        solver.collectResults();

        helper.finalize();
    } catch (std::string &e) {
        std::cerr << "CGM Error: " << e << std::endl;
        MPIHelper::Abort(1);
        return 1;
    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        MPIHelper::Abort(2);
        return 2;
    } catch (...) {
        std::cerr << "Unknown Error!!!" << std::endl;
        MPIHelper::Abort(3);
        return 3;
    }
    return 0;
}
