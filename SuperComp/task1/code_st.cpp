#include <iostream>
#include <fstream>

#include <boost/graph/page_rank.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/algorithm/string.hpp>

using namespace boost;

typedef adjacency_list<vecS, vecS, directedS> Graph;

void read_graph(const std::string &file_name, Graph &g, unsigned int &vertices_count, long long &edges_count) {
    std::fstream file(file_name.c_str(), std::fstream::in | std::fstream::binary);

    file.read((char*)(&vertices_count), sizeof(int));
    file.read((char*)(&edges_count), sizeof(long long));

    // add edges from file
    for(long long i = 0; i < edges_count; i++) {
        unsigned int src_id = 0, dst_id = 0;
        float weight = 0;

        // read i-th edge data
        file.read((char*)(&src_id), sizeof(int));
        file.read((char*)(&dst_id), sizeof(int));

        //print edge data
        add_edge(vertex(src_id, g), vertex(dst_id, g), g);
    }

    file.close();
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Параметры: <входной файл> <масштаб графа>" << std::endl;
        return 1;
    }

    // получение имени входного файла с графом, числа вершин входного графа, а так же количество итераций pagerank
    std::string file_name = argv[1];
    unsigned int vertices_count = 1U << strtol(argv[2], NULL, 10);
    long long edges_count = 0;

    // Создаем граф с заданным числом вершин
    Graph g(vertices_count);

    // читаем граф на корневом процессе
    read_graph(file_name, g, vertices_count, edges_count);

    // считаем PageRank с замером времени выполнения
    std::vector<double> ranks(num_vertices(g));

    graph::page_rank(g, make_iterator_property_map(
            ranks.begin(), get(vertex_index, g)));

    std::vector<std::string> strs;
    split(strs, file_name, is_any_of("/"));
    std::string fname = "../res/" + strs[strs.size()-1] + ".st_res";
    std::ofstream out_file;

    out_file.open(fname.c_str());
    out_file << ranks.size() << std::endl;
    for (int i = 0; i < ranks.size(); ++i) {
        out_file << ranks[i] << " ";
    }

    return 0;
}
