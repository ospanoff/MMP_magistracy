#include <mpi.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int processes;
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char data[1024];
    MPI_Status status;

    if (rank != 0) {
        const char *message = "Some message";
        MPI_Send(message, strlen(message) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    } else {
        for (int i = 1; i < processes; i++) {
            MPI_Recv(&data, 100, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            int data_len;
            MPI_Get_count(&status, MPI_CHAR, &data_len);
            data[data_len] = 0;

            printf("Received '%s' from process #%d\n", data, status.MPI_SOURCE);
        }
    }

    MPI_Finalize();
}
