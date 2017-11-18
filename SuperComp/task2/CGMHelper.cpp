//
// Created by Ayat ".ospanoff" Ospanov
//

#include <cmath>

#include "CGMHelper.h"


////////////////////////////////////////////////////////////////////////////////
/// Func2D
////////////////////////////////////////////////////////////////////////////////
Func2D::Func2D(int size_x_, int size_y_)
        :sizeX(size_x_), sizeY(size_y_) {
    resize(sizeX, sizeY);
}

double &Func2D::operator()(int x, int y) {
    return f[y * sizeX + x];
}

double Func2D::operator()(int x, int y) const {
    return f[y * sizeX + x];
}

void Func2D::resize(int size_x, int size_y) {
    sizeX = size_x;
    sizeY = size_y;
    f.resize(1u * sizeX * sizeY);
    std::fill(f.begin(), f.end(), 0.);
}


////////////////////////////////////////////////////////////////////////////////
/// Grid1D
////////////////////////////////////////////////////////////////////////////////
double Grid1D::f(double t, float q) {
    return (std::pow(static_cast<float>(1.0 + t), q) - 1.0) / (std::pow(2.0f, q) - 1.0);
}

double Grid1D::get_coord(int i) {
    double t = 1.0 * i / (numOfPts - 1);
    return A2 * f(t) + A1 * (1 - f(t));
}

void Grid1D::init(int begin, int end) {
    grid.resize(static_cast<unsigned int>(end - begin));
    for (int i = begin; i < end; ++i) {
        grid[i - begin] = get_coord(i);
    }
}

double Grid1D::step(int i) const {
    return grid[i + 1] - grid[i];
}

double Grid1D::midStep(int i) const {
    return 0.5 * (step(i) + step(i - 1));
}


////////////////////////////////////////////////////////////////////////////////
/// Exchanger
////////////////////////////////////////////////////////////////////////////////
void Exchanger::exchange(Func2D &data) {
    sendData.clear();
    for (int x = sendPart.startX; x < sendPart.endX; ++x) {
        for (int y = sendPart.startY; y < sendPart.endY; ++y) {
            sendData.push_back(data(x, y));
        }
    }

    helper.Isend(sendData, exchangeRank, sendReq);
    helper.Irecv(recvData, exchangeRank, recvReq);
}

void Exchanger::wait(Func2D &data) {
    helper.Wait(recvReq);

    std::vector<double>::const_iterator value = recvData.begin();
    for (int x = recvPart.startX; x < recvPart.endX; ++x) {
        for (int y = recvPart.startY; y < recvPart.endY; ++y) {
            data(x, y) = *(value++);
        }
    }

    helper.Wait(sendReq);
}
