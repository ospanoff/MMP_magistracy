#ifndef cpts
#define cpts

#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>


//-------------------------------------------------
//-------------------------------------------------
// Point
//-------------------------------------------------
//-------------------------------------------------
class Point {
public:
    std::uint32_t id;
    std::double_t x, y;

    explicit Point(std::double_t x = 0, std::double_t y = 0)
            :id(0),x(x),y(y) {}

    bool is_less(Point &p, int axis=0) {
        if (axis == 0) {
            return x < p.x || (x == p.x && y < p.y);
        } else {
            return y < p.y;
        }
    }

    std::double_t count_dist(Point &p) {
        return std::fabs(x - p.x) + std::fabs(y - p.y);
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
    std::double_t min_dist = 1e20;
    std::tuple<Point, Point> closest_pts;
public:
    explicit ClosestPointsFinder(std::vector<Point> _pts)
            :pts(std::move(_pts)) {
        tmp.resize(pts.size());
    };

    void step(std::uint32_t l, std::uint32_t r);
    void brute_force(std::uint32_t l, std::uint32_t r);
    void update_dist(Point &p1, Point &p2);

    void find_2closest_points() {
        merge_sort(pts, 0, pts.size());
        step(0, pts.size());
    };

    // getters
    std::tuple<Point, Point> get_2closest_points() {
        return closest_pts;
    };

    std::double_t get_min_dist() {
        return min_dist;
    }

    std::string get_fancy_results() {
        Point p1, p2;
        std::tie(p1, p2) = closest_pts;
        char buf[2048];
        int n = 0;
        n = sprintf(buf, "Номера двух ближайших точек: %d и %d\n", p1.id, p2.id);
        n += sprintf(buf + n, "Точка %d: (%lf, %lf)\n", p1.id, p1.x, p1.y);
        n += sprintf(buf + n, "Точка %d: (%lf, %lf)\n", p2.id, p2.x, p2.y);
        sprintf(buf + n, "Расстояние: %lf\n", min_dist);
        return std::string(buf);
    }
};

#endif
