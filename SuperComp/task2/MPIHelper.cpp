//
// Created by ospanoff on 11/13/17.
//

#include <string>

#include "MPIHelper.h"
#include "CGMHelper.h"


////////////////////////////////////////////////////////////////////////////////
/// MPIHelper
////////////////////////////////////////////////////////////////////////////////

bool MPIHelper::initialized = false;

MPIHelper::MPIHelper() {
    rank = 0;
    numOfProcesses = 0;
}

void MPIHelper::Init(int *argc, char ***argv) {
    if (initialized) {
        throw CGMException("MPI has already been initialized!!!");
    }
    Check(MPI_Init(argc, argv), "MPI_Init");
    Check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    Check(MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses), "MPI_Comm_size");
#ifdef USE_OMP
    numOfOMPThreads = omp_get_max_threads();
#endif
    initialized = true;
}

void MPIHelper::Finalize() {
    if (!initialized) {
        throw CGMException("MPI has not been initialized");
    }
    MPI_Finalize();
}

void MPIHelper::Isend(std::vector<double> &data, int rank, MPI_Request &request) {
    Check(MPI_Isend(data.data(), data.size(), MPI_DOUBLE,
                    rank, 0, MPI_COMM_WORLD, &request),
          "MPI_Isend");
}

void MPIHelper::Irecv(std::vector<double> &data, int rank, MPI_Request &request) {
    Check(MPI_Irecv(data.data(), data.size(), MPI_DOUBLE,
                    rank, 0, MPI_COMM_WORLD, &request),
          "MPI_Irecv");
}

void MPIHelper::Wait(MPI_Request &request) {
    Check(MPI_Wait(&request, MPI_STATUS_IGNORE), "MPI_Wait");
}

void MPIHelper::AllReduce(std::vector<double> &data) {
    Check(MPI_Allreduce(MPI_IN_PLACE, data.data(), data.size(),
                        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
          "MPI_Allreduce");
}

void MPIHelper::Gather(int &x, std::vector<int> &data) {
    Check(MPI_Gather(&x, 1, MPI_INT, data.data(), 1, MPI_INT, 0, MPI_COMM_WORLD),
          "MPI_Gather");
}

void MPIHelper::Gatherv(std::vector<double> &send, std::vector<double> &recv, std::vector<int> &size, std::vector<int> &offset) {
    Check(MPI_Gatherv(send.data(), send.size(), MPI_DOUBLE, recv.data(), size.data(), offset.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD),
          "MPI_Gatherv");
}

void MPIHelper::ReduceMax(double x, double &res) {
    Check(MPI_Reduce(&x, &res, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), "MPI_Reduce");
}

double MPIHelper::Time() {
    Check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
    return MPI_Wtime();
}

void MPIHelper::Check(const int mpiResult, const char *mpiFunctionName) {
    if (mpiResult != MPI_SUCCESS) {
        char s[1024];
        sprintf(s, "Function '%s' has failed with code %d", mpiFunctionName, mpiResult);
        throw CGMException(s);
    }
}

void MPIHelper::Abort(int code) {
    if (initialized) {
        MPI_Abort(MPI_COMM_WORLD, code);
    }
}