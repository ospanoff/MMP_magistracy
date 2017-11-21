#include <iostream>
#include <vector>
#include <map>

#include <chrono>

#include "utils.cpp"

int main(int argc, char const *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: ./kruskal <file name>" << std::endl;
        return 1;
    }

    Graph g;
    g.read_graph(argv[1]);
    auto start = std::chrono::high_resolution_clock::now();
    g.sort_edges();

    Graph mst(g.get_names());

    DSU dsu(g.get_vertex_count());

    for (int i = 0; i < g.get_edges_count(); ++i) {
        int a = g[i].a, b = g[i].b;
        if (dsu.find_set(a) != dsu.find_set(b)) {
            mst.add_edge(g[i]);
            dsu.union_sets(a, b);
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s.\n";
    std::cout << "|E| = " << g.get_edges_count() << "; |V| = " << g.get_vertex_count() << std::endl;

    mst.write_graph("output.txt");

    return 0;
}
