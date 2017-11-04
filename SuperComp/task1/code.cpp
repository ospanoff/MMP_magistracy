#include <boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/distributed/adjacency_list.hpp>
#include <boost/graph/metis.hpp>
#include <boost/graph/distributed/graphviz.hpp>
#include <fstream>
#include <string>
#include <iostream>

using namespace boost;

using boost::graph::distributed::mpi_process_group;

typedef adjacency_list<vecS, distributedS<mpi_process_group, vecS>, undirectedS,
/*Vertex properties=*/property<vertex_distance_t, float>,
/*Edge properties=*/property<edge_weight_t, float> >
Graph;

int main(int argc, char* argv[]) {
    // Инициализация MPI среды
    boost::mpi::environment env(argc,argv);


    // получение числа вершин входного графа, имени входного файла с графом, а так же имени выходного файла с гарфом
    int vertices_pow = atoi(argv[1]);
    int vertices_count = pow(2.0, vertices_pow);
    long long edges_count = 0;
    std::string file_name = argv[2];

    // Создаем граф с заданным числом вершин
    Graph g(vertices_count);

    // читаем граф из на корневом процессе и добавляем в него ребра
    // if (process_id(process_group(g)) == 0) {
    //     std::fstream file(file_name, std::ios::in | std::ios::binary);
    //
    //     vertices_count = 0;
    //     edges_count = 0;
    //     file.read((char*)(&vertices_count), sizeof(int));
    //     file.read((char*)(&edges_count), sizeof(long long));
    //
    //     // add edges from file
    //     for(int i = 0; i < edges_count; i++) {
    //         int src_id = 0, dst_id = 0;
    //         float weight = 0;
    //
    //         // read i-th edge data
    //         file.read((char*)(&src_id), sizeof(int));
    //         file.read((char*)(&dst_id), sizeof(int));
    //         file.read((char*)(&weight), sizeof(float)); // remove it for unweighed graph
    //
    //         //print edge data
    //         add_edge(vertex(src_id, g), vertex(dst_id, g), weight, g);
    //     }
    //
    //     file.close();
    // }

    // выбираем нулевую вершину графа
    // graph_traits<Graph>::vertex_descriptor start = vertex(0, g);

    // // считаем кратчайшие пути с замером времени выполнения
    // double t1 = MPI_Wtime();
    // MPI_Barrier(MPI_COMM_WORLD);
    // dijkstra_shortest_paths(g, start, distance_map(get(vertex_distance, g)));
    // MPI_Barrier(MPI_COMM_WORLD);
    // double t2 = MPI_Wtime();
    //
    // // печатаем производительность с корневого процесса
    // if (process_id(process_group(g)) == 0) {
    //     std::cout << "Performance: " << edges_count / ((t2 -
    //     t1) * 1e6) << " MTEPS" << std::endl << std::endl;
    //     std::cout << "Time: " << t2 - t1 << " seconds" << std::endl;
    // }
    //
    // // если указано имя выходного файла, сохраняем туда граф с подсчитанными параллельно расстояниями в формате graphviz
    // if (argc > 3) {
    //     std::string outfile = argv[3];
    //     if (process_id(process_group(g)) == 0) {
    //         std::cout << "Writing GraphViz output to " << outfile << "... ";
    //         std::cout.flush();
    //     }
    //     write_graphviz(outfile, g,
    //     make_label_writer(get(vertex_distance, g)),
    //     make_label_writer(get(edge_weight, g)));
    //     if (process_id(process_group(g)) == 0)
    //     std::cout << "Done." << std::endl;
    // }
    return 0;
}
