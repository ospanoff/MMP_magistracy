#ifndef cpts
#define cpts

#include <iostream>
#include <vector>
#include <tuple>


//-------------------------------------------------
//-------------------------------------------------
// Tools
//-------------------------------------------------
//-------------------------------------------------
std::uint64_t abs_llu_diff(std::uint64_t a, std::uint64_t b);


//-------------------------------------------------
//-------------------------------------------------
// Point
//-------------------------------------------------
//-------------------------------------------------
class Point {
public:
    std::uint32_t id;
    std::uint64_t x, y;

    explicit Point(std::uint64_t x=0, std::uint64_t y=0)
            :id(0),x(x),y(y) {}

    bool is_less(Point &p, int axis=0) {
        if (axis == 0) {
            return x < p.x || (x == p.x && y < p.y);
        } else {
            return y < p.y;
        }
    }

    std::uint64_t count_dist(Point &p) {
        return abs_llu_diff(x, p.x) + abs_llu_diff(y, p.y);
    }
};
std::ostream &operator<<(std::ostream &os, Point const &p);


//-------------------------------------------------
//-------------------------------------------------
// Merge Sort
//-------------------------------------------------
//-------------------------------------------------
void merge(std::vector<Point> &pts, std::uint32_t l, std::uint32_t r, std::uint32_t m, int axis);
void merge_sort(std::vector<Point> &pts, std::uint32_t l, std::uint32_t r, int axis=0);


//-------------------------------------------------
//-------------------------------------------------
// ClosestPointsFinder
//-------------------------------------------------
//-------------------------------------------------
class ClosestPointsFinder {
    std::vector<Point> pts;
    std::vector<Point> tmp;
    std::uint64_t min_dist = (std::uint64_t)1 << 63;
    std::tuple<Point, Point> closest_pts;
public:
    explicit ClosestPointsFinder(std::vector<Point> _pts)
            :pts(std::move(_pts)) {
        tmp.resize(pts.size());
    };

    void step(std::uint32_t l, std::uint32_t r);
    void update_dist(Point &p1, Point &p2);

    void find_2closest_points() {
        merge_sort(pts, 0, pts.size());
        step(0, pts.size());
    };

    // getters
    std::tuple<Point, Point> get_2closest_points() {
        return closest_pts;
    };

    std::uint64_t get_min_dist() {
        return min_dist;
    }

    std::string get_fancy_results() {
        Point p1, p2;
        std::tie(p1, p2) = closest_pts;
        char buf[2048];
        int n = 0;
        n = sprintf(buf, "Номера двух ближайших точек: %d и %d\n", p1.id, p2.id);
        n += sprintf(buf + n, "Точка 1: (%ld, %ld)\n", p1.x, p1.y);
        n += sprintf(buf + n, "Точка 2: (%ld, %ld)\n", p2.x, p2.y);
        sprintf(buf + n, "Расстояние: %ld\n", min_dist);
        return std::string(buf);
    }
};

#endif
