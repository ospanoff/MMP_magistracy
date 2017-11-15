#include <iostream>

#include <boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include <boost/graph/distributed/page_rank.hpp>
#include <boost/algorithm/string.hpp>

#include "graph.cpp"

using namespace boost;
using boost::graph::distributed::mpi_process_group;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Параметры: <входной файл> <масштаб графа>" << std::endl;
        return 1;
    }

    // Инициализация MPI среды
    mpi::environment env(argc,argv);
    boost::mpi::communicator world;

    // получение имени входного файла с графом, числа вершин входного графа
    std::string file_name = argv[1];
    unsigned int vertices_count = 1U << strtol(argv[2], NULL, 10);
    long long edges_count = 0;

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
    graph::page_rank(g, make_iterator_property_map(ranks.begin(), get(vertex_index, g)));
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    // печатаем производительность с корневого процесса
    if (process_id(process_group(g)) == 0) {
        std::cout << "Performance: " << 1.0 * edges_count / ((t2 - t1) * 1e3) << " KTEPS" << std::endl;
        std::cout << "Time: " << t2 - t1 << " seconds" << std::endl;

        std::vector<std::vector<double> > all_ranks;

        mpi::gather(world, ranks, all_ranks, 0);

        std::vector<std::string> strs;
        split(strs, file_name, is_any_of("/"));
        std::string fname = "../res/" + strs[strs.size()-1] + "_" + lexical_cast<std::string>(process_group(g).size) + ".mp_res";
        std::ofstream out_file;

        int size = 0;
        for (int i = 0; i < all_ranks.size(); ++i) {
            size += all_ranks[i].size();
        }
        out_file.open(fname.c_str());
        out_file << size << std::endl;
        for (int i = 0; i < all_ranks.size(); ++i) {
            for (int j = 0; j < all_ranks[i].size(); ++j) {
                out_file << all_ranks[i][j] << " ";
            }
        }
    } else {
        mpi::gather(world, ranks, 0);
    }

    return 0;
}
