#include "closest_points.h"


//-------------------------------------------------
//-------------------------------------------------
// Point
//-------------------------------------------------
//-------------------------------------------------
std::ostream &operator<<(std::ostream &os, Point const &p) {
    return os << p.id << ": (" << p.x << ", " << p.y << ")" << std::endl;
}


//-------------------------------------------------
//-------------------------------------------------
// Merge Sort
//-------------------------------------------------
//-------------------------------------------------
void merge(std::vector<Point> &pts,
           std::uint32_t l, std::uint32_t r, std::uint32_t m,
           int axis) {
    std::vector<Point> L(pts.begin() + l, pts.begin() + m),
                       R(pts.begin() + m, pts.begin() + r);

    std::uint32_t i = 0, j = 0;
    for (std::uint32_t k = l; k < r; ++k) {
        if (i < L.size() && j < R.size() && L[i].is_less(R[j], axis)) {
            pts[k] = L[i++];
        } else if (j < R.size()) {
            pts[k] = R[j++];
        } else {
            pts[k] = L[i++];
        }
    }
}

void merge_sort(std::vector<Point> &pts, std::uint32_t l, std::uint32_t r, int axis) {
    if (l < r - 1) {
        std::uint32_t m = (l + r) / 2;
        merge_sort(pts, l, m, axis);
        merge_sort(pts, m, r, axis);
        merge(pts, l, r, m, axis);
    }
}


//-------------------------------------------------
//-------------------------------------------------
// ClosestPointsFinder
//-------------------------------------------------
//-------------------------------------------------
void ClosestPointsFinder::update_dist(Point &p1, Point &p2) {
    auto dist = p1.count_dist(p2);
    if (dist < min_dist) {
        min_dist = dist;
        closest_pts = std::make_tuple(p1, p2);
    }
}

void ClosestPointsFinder::brute_force(std::uint32_t l, std::uint32_t r) {
    for (std::uint32_t i = l; i < r; ++i) {
        for (std::uint32_t j = i + 1; j < r; ++j) {
            update_dist(pts[i], pts[j]);
        }
    }
}

void ClosestPointsFinder::step(std::uint32_t l, std::uint32_t r) {
    if (r - l < 4) {
        brute_force(l, r);
        merge_sort(pts, l, r, 1);
        return;
    }

    std::uint32_t m = (l + r) / 2;
    step(l, m);
    step(m, r);
    merge(pts, l, r, m, 1);

    std::uint32_t k = 0;
    for (std::uint32_t i = l; i < r; ++i) {
        if (std::fabs(pts[i].x - pts[m].x) < min_dist) {
            for (std::int32_t j = k - 1; j >= 0 && pts[i].y - tmp[j].y < min_dist; --j) {
                update_dist(pts[i], tmp[j]);
            }
            tmp[k++] = pts[i];
        }
    }

}
