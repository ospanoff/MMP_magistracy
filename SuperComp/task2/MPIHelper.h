//
// Created by Ayat ".ospanoff" Ospanov
//

#ifndef MPIHELPER_H
#define MPIHELPER_H


#include <mpi.h>
#include <string>
#include <vector>


class MPIHelper {
private:
    MPI_Comm comm;
    int numOfProcesses;
    int numOfProcs[2];
    int coords[2];
    int numOfOMPThreads;

public:
    static MPIHelper &getInstance() {
        static MPIHelper helper;
        return helper;
    }

    void init(int *argc, char ***argv);
    void finalize();

    MPI_Comm getComm() const;

    int getRank() const;
    int getRank(int rankX, int rankY) const;
    int getRankX() const;
    int getRankY() const;

    int getNumOfProcs() const;
    int getNumOfProcsX() const;
    int getNumOfProcsY() const;

    int getNumOfOMPThreads() const;

    bool isMaster() const;
    bool hasLeftNeighbour() const { return getRankX() > 0; }
    bool hasRightNeighbour() const { return getRankX() < getNumOfProcsX() - 1; }
    bool hasTopNeighbour() const { return getRankY() > 0; }
    bool hasBottomNeighbour() const { return getRankY() < getNumOfProcsY() - 1; }

    static void Check(int mpiResult, const char *mpiFunctionName);
    static void Abort(int code);

    void AllReduceSum(double &data);
    void Isend(std::vector<double> &data, int rank, MPI_Request &request);
    void Irecv(std::vector<double> &data, int rank, MPI_Request &request);
    void GatherInts(int x, std::vector<int> &data);
    void Gatherv(std::vector<double> &send, std::vector<double> &recv,
                 std::vector<int> &sizes, std::vector<int> &offsets);
    void Wait(MPI_Request &request);

    double time();

private:
    MPIHelper() {}
    MPIHelper(MPIHelper const&);
    void operator=(MPIHelper const&);
};


#endif //MPIHELPER_H
