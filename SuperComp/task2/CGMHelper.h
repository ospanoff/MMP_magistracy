//
// Created by Ayat ".ospanoff" Ospanov
//

#ifndef CGM_HELPER_H
#define CGM_HELPER_H

#include <vector>
#include <sstream>

#include "MPIHelper.h"

#define to_string(x) dynamic_cast<std::ostringstream &>(std::ostringstream() << std::dec << x).str()


////////////////////////////////////////////////////////////////////////////////
/// CGMException
////////////////////////////////////////////////////////////////////////////////
class CGMException : public std::exception {
    std::string msg;
public:
    explicit CGMException(const char *errorMsg = "") : msg(errorMsg) {}
    virtual ~CGMException() throw() {}
    virtual const char *what() const throw() {
        return msg.c_str();
    }
};


////////////////////////////////////////////////////////////////////////////////
/// CGMResult
////////////////////////////////////////////////////////////////////////////////
class CGMResult {
public:
    int rank;
    double timeElapsed;
    int iterations;

    explicit CGMResult(int rank = 0, double time = 0, int iter = 0) : rank(rank),timeElapsed(time),iterations(iter) {}
    void print() {
        printf("Rank: %d\nElapsed time: %lf; Iters: %d\n", rank, timeElapsed, iterations);
    }
};

////////////////////////////////////////////////////////////////////////////////
/// Fraction
////////////////////////////////////////////////////////////////////////////////
class Fraction {
public:
    double numerator;
    double denominator;

    Fraction(double n, double d) : numerator(n),denominator(d) {}
    double divide() { return numerator / denominator; }
};


////////////////////////////////////////////////////////////////////////////////
/// Func2D
////////////////////////////////////////////////////////////////////////////////
class Func2D {
public:
    std::vector<double> f;
    int sizeX;
    int sizeY;
    Func2D() : sizeX(0),sizeY(0) {}
    Func2D(int size_x, int size_y);

    void resize(int size_x, int size_y);
    int size() { return sizeX * sizeY; }

    double &operator()(int x, int y);
    double operator()(int x, int y) const;
};


////////////////////////////////////////////////////////////////////////////////
/// Grid1D
////////////////////////////////////////////////////////////////////////////////
class Grid1D {
    std::vector<double> grid;
    double A1;
    double A2;
    int numOfPts;

public:
    Grid1D(double A1, double A2, int numOfPts)
            :A1(A1),A2(A2),numOfPts(numOfPts) {};

    double f(double t, float q=3.f/2);
    double get_coord(int i);

    void init(int begin, int end);

    double &operator[](int i) {
        return grid[i];
    }
    double operator[](int i) const {
        return grid[i];
    }
    int size() const {
        return static_cast<int>(grid.size());
    }

    double step(int i) const;
    double midStep(int i) const;
};


////////////////////////////////////////////////////////////////////////////////
/// Exchanger
////////////////////////////////////////////////////////////////////////////////
class EdgeIndexer {
public:
    int startX, endX;
    int startY, endY;

public:
    enum {
        ByX,
        ByY
    };
    EdgeIndexer(int dir, int dirStart, const Grid1D &perpGrid) {
        switch (dir) {
            case ByX:
                startX = dirStart;
                endX = dirStart + 1;
                startY = 1;
                endY = perpGrid.size() - 1;
                break;

            case ByY:
                startX = 1;
                endX = perpGrid.size() - 1;
                startY = dirStart;
                endY = dirStart + 1;
                break;

            default:
                break;
        }
    }

    unsigned long size() {
        return 1u * (endX - startX) * (endY - startY);
    }
};


class Exchanger {
private:
    MPIHelper helper;

    int exchangeRank;

    EdgeIndexer sendPart;
    std::vector<double> sendData;
    MPI_Request sendReq;

    EdgeIndexer recvPart;
    std::vector<double> recvData;
    MPI_Request recvReq;

public:
    Exchanger(MPIHelper helper, int rank, EdgeIndexer send, EdgeIndexer recv)
            :helper(helper),exchangeRank(rank),
             sendPart(send),recvPart(recv) {
        sendData.reserve(sendPart.size());
        recvData.resize(recvPart.size());
    }

    void exchange(Func2D &data);
    void wait(Func2D &data);
};


#endif //CGM_HELPER_H
