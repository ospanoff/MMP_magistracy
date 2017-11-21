#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <fstream>
#include <iostream>

class Edge {
public:
    double weight;

    unsigned long a;
    std::string a_name;

    unsigned long b;
    std::string b_name;

public:
    Edge() {}
    Edge(double w, int in, int out)
        :weight(w),a(in),b(out) {}

    bool operator<(const Edge &e) {
        return weight < e.weight;
    }
};

class Graph {
private:
    std::vector<Edge> edges;
    std::map<std::string, unsigned long> names;
    unsigned long vertex_cnt;
    double length;

public:
    Graph() {
        vertex_cnt = 0;
        length = 0;
    }

    Graph(std::map<std::string, unsigned long> names):names(names) {}

    void add_edge(Edge e) {
        e.a = get_id(e.a_name);
        e.b = get_id(e.b_name);
        length += e.weight;
        edges.push_back(e);
    }

    void sort_edges() {
        sort(edges.begin(), edges.end());
    }

    int get_id(std::string name) {
        auto it = names.find(name);
        if (it != names.end()) {
            return it->second;
        } else {
            names[name] = vertex_cnt;
            return vertex_cnt++;
        }
    }

    int get_edges_count() {
        return edges.size();
    }

    int get_vertex_count() {
        return vertex_cnt;
    }

    std::map<std::string, unsigned long> get_names() {
        return names;
    }

    void read_graph(std::string fname) {
        std::ifstream file;
        file.open(fname);
        std::string a, b;
        double w;
        Edge e;
        while (file >> a >> b >> w) {
            e.a_name = a;
            e.b_name = b;
            e.weight = w;
            add_edge(e);
        }
        file.close();
    }

    void write_graph(std::string fname) {
        std::ofstream file;
        file.open(fname);
        file << length << std::endl;
        for (auto i = 0u; i < edges.size(); ++i) {
            char s[1024];
            auto e = edges[i];
            sprintf(s, "%s [%ld] - %s [%ld]", e.a_name.c_str(), e.a, e.b_name.c_str(), e.b);
            file << s << std::endl;
        }
        file.close();
    }

    Edge operator[](int i) {
        return edges[i];
    }
};

class DSU {
    std::vector<int> parent;
public:
    DSU(int n) {
        parent.resize(n);
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
        }
    }

    int find_set(int v) {
        if (v == parent[v])
    		return v;
    	return parent[v] = find_set(parent[v]);
    }

    void union_sets(int a, int b) {
        a = find_set(a);
    	b = find_set(b);
    	if (a != b)
    		parent[a] = b;
    }
};
