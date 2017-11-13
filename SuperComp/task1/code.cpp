#include <iostream>

#include <boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include <boost/graph/page_rank.hpp>

#include "graph.cpp"

using namespace boost;
using boost::graph::distributed::mpi_process_group;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "Параметры: <входной файл> <масштаб графа> <количество итераций>" << std::endl;
        return 1;
    }

    // Инициализация MPI среды
    mpi::environment env(argc,argv);

    // получение имени входного файла с графом, числа вершин входного графа, а так же количество итераций pagerank
    std::string file_name = argv[1];
    unsigned int vertices_count = 1U << strtol(argv[2], NULL, 10);
    long long edges_count = 0;
    size_t iters = strtoul(argv[3], NULL, 10);

    // Создаем граф с заданным числом вершин
    Graph g(vertices_count);

    // читаем граф на корневом процессе
    if (process_id(process_group(g)) == 0) {
        read_graph(file_name, g, vertices_count, edges_count);
    }

    // считаем PageRank с замером времени выполнения
    std::vector<double> ranks(num_vertices(g));

    double t1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    page_rank(g, make_iterator_property_map(ranks.begin(), get(vertex_index, g)),
              graph::n_iterations(iters), 0.85, vertices_count);
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    // печатаем производительность с корневого процесса
    if (process_id(process_group(g)) == 0) {
        std::cout << "Performance: " << 1.0 * edges_count / ((t2 - t1) * 1e3) << " KTEPS" << std::endl;
        std::cout << "Time: " << t2 - t1 << " seconds" << std::endl;
    }

    return 0;
}
