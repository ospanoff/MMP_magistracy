//
// Created by ospanoff on 11/13/17.
//

#ifndef TASK2_MPIHELPER_H
#define TASK2_MPIHELPER_H

#include <omp.h>
#include <mpi.h>
#include <vector>


class MPIHelper {
public:
    int rank;
    int numOfProcesses;
    int numOfOMPThreads;
    static bool initialized;

public:
    MPIHelper();

    void Init(int *argc, char ***argv);
    void Finalize();

    void Isend(std::vector<double> &data, int rank, MPI_Request &request);
    void Irecv(std::vector<double> &data, int rank, MPI_Request &request);
    void Wait(MPI_Request &request);
    void AllReduce(std::vector<double> &data);
    void Gather(int &x, std::vector<int> &data);
    void Gatherv(std::vector<double> &send, std::vector<double> &recv, std::vector<int> &size, std::vector<int> &offset);
    void ReduceMax(double x, double &res);

    static double Time();
    static void Check(int mpiResult, const char *mpiFunctionName);
    static void Abort(int code);
};


#endif //TASK2_MPIHELPER_H
