//
// Created by Ayat ".ospanoff" Ospanov
//

#include <omp.h>

#include "MPIHelper.h"


void MPIHelper::init(int *argc, char ***argv) {
    Check(MPI_Init(argc, argv), "MPI_Init");

    Check(MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses), "MPI_Comm_size");

    int pow = 0;
    int nProc = 1;
    while (nProc < numOfProcesses) {
        nProc *= 2;
        ++pow;
    }
    if (nProc != numOfProcesses) {
        throw std::string("Number of processes should be the power of 2");
    }
    numOfProcs[0] = 1 << (pow / 2);
    numOfProcs[1] = 1 << (pow - pow / 2);

    int periodic[2] = {0};
    Check(MPI_Cart_create(MPI_COMM_WORLD, 2, numOfProcs, periodic, 1, &comm), "MPI_Cart_create");

    int rank;
    Check(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank");
    Check(MPI_Cart_coords(comm, rank, 2, coords), "MPI_Cart_coords");

#ifdef USE_OMP
    numOfOMPThreads = omp_get_max_threads();
#endif
}

void MPIHelper::finalize() {
    MPI_Finalize();
}

bool MPIHelper::isMaster() const {
    return (coords[0] == 0) && (coords[1] == 0);
}

int MPIHelper::getRank() const {
    int rank;
    Check(MPI_Cart_rank(comm, coords, &rank), "MPI_Cart_rank");
    return rank;
}

int MPIHelper::getRank(int rankX, int rankY) const {
    int rank;
    int crds[] = { rankX, rankY };
    Check(MPI_Cart_rank(comm, crds, &rank), "MPI_Cart_rank");
    return rank;
}

int MPIHelper::getRankX() const {
    return coords[0];
}

int MPIHelper::getRankY() const {
    return coords[1];
}

int MPIHelper::getNumOfProcsX() const {
    return numOfProcs[0];
}

int MPIHelper::getNumOfProcsY() const {
    return numOfProcs[1];
}

int MPIHelper::getNumOfProcs() const {
    return numOfProcesses;
}

int MPIHelper::getNumOfOMPThreads() const {
    return numOfOMPThreads;
}

void MPIHelper::Check(int mpiResult, const char *mpiFunctionName) {
    if (mpiResult != MPI_SUCCESS) {
        char s[1024];
        sprintf(s, "Function '%s' has failed with code %d", mpiFunctionName, mpiResult);
        throw std::string(s);
    }
}

void MPIHelper::Abort(int code) {
    MPIHelper &helper = MPIHelper::getInstance();
    MPI_Abort(helper.comm, code);
}

void MPIHelper::AllReduceSum(double &data) {
    Check(MPI_Allreduce(MPI_IN_PLACE, &data, 1, MPI_DOUBLE, MPI_SUM, comm),
          "MPI_Allreduce");
}

void MPIHelper::Isend(std::vector<double> &data, int rank, MPI_Request &request) {
    Check(MPI_Isend(data.data(), data.size(), MPI_DOUBLE,
                    rank, 0, comm, &request),
          "MPI_Isend");
}

void MPIHelper::Irecv(std::vector<double> &data, int rank, MPI_Request &request) {
    Check(MPI_Irecv(data.data(), data.size(), MPI_DOUBLE,
                    rank, 0, comm, &request),
          "MPI_Irecv");
}

void MPIHelper::GatherInts(int x, std::vector<int> &data) {
    Check(MPI_Gather(&x, 1, MPI_INT, data.data(), 1, MPI_INT, 0, comm),
          "MPI_Gather");
}

void MPIHelper::Gatherv(std::vector<double> &send, std::vector<double> &recv,
                        std::vector<int> &sizes, std::vector<int> &offsets) {
    Check(MPI_Gatherv(send.data(), send.size(), MPI_DOUBLE, recv.data(), sizes.data(), offsets.data(), MPI_DOUBLE, 0, comm),
          "MPI_Gatherv");
}

void MPIHelper::Wait(MPI_Request &request) {
    Check(MPI_Wait(&request, MPI_STATUS_IGNORE), "MPI_Wait");
}

double MPIHelper::time() {
    Check(MPI_Barrier(comm), "MPI_Barrier");
    return MPI_Wtime();
}
