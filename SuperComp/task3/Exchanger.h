//
// Created by Ayat ".ospanoff" Ospanov
//

#ifndef EXCHANGER_H
#define EXCHANGER_H


#include <mpi.h>
#include <vector>


class EdgeIndexer {
public:
    int startX, endX;
    int startY, endY;

public:
    enum {
        ByX,
        ByY
    };
    EdgeIndexer(int dir, int dirStart, int perpendSize) {
        switch (dir) {
            case ByX:
                startX = dirStart;
                endX = dirStart + 1;
                startY = 1;
                endY = perpendSize - 1;
                break;

            case ByY:
                startX = 1;
                endX = perpendSize - 1;
                startY = dirStart;
                endY = dirStart + 1;
                break;

            default:
                break;
        }
    }

    unsigned int size() {
        return 1u * (endX - startX) * (endY - startY);
    }
};


class Exchanger {
public:
    int exchangeRank;

    EdgeIndexer sendPart;
    std::vector<double> sendData;
    MPI_Request sendReq;

    EdgeIndexer recvPart;
    std::vector<double> recvData;
    MPI_Request recvReq;

public:
    Exchanger(int rank, EdgeIndexer send, EdgeIndexer recv)
            :exchangeRank(rank),sendPart(send),recvPart(recv) {
        sendData.reserve(sendPart.size());
        recvData.resize(recvPart.size());
    }
};


#endif //EXCHANGER_H
