#include <boost/graph/distributed/adjacency_list.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include <fstream>

using namespace boost;
using boost::graph::distributed::mpi_process_group;

typedef adjacency_list<vecS, distributedS<mpi_process_group, vecS>, bidirectionalS,
/*Vertex properties=*/no_property, //property<vertex_distance_t, float>,
/*Edge properties=*/property<edge_weight_t, float> >
Graph;

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
        file.read((char*)(&weight), sizeof(float));

        //print edge data
        add_edge(vertex(src_id, g), vertex(dst_id, g), weight, g);
    }

    file.close();
}
