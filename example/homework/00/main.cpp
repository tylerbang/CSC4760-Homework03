/*
Write a program that uses MPI Comm split to create two sub-communicators for each process in the world
of processes given initially in MPI COMM WORLD. The first split should put processes together that have the
same color when their ranks are divided by an integer Q (ranks in MPI COMM WORLD). The second split should
put processes together that have the same color when your compute the color as their rank mod Q. In this
situation, your world size must be at exactly P × Q, P, Q ≥ 1. You get to pick P , Q but P × Q has to be
the size of the process group of MPI COMM WORLD.
*/

#include <mpi.h>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int P = 2, Q = size / P;
    if (size % P != 0){
        cout << "P must be a factor of the number of processes" << endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Comm color_comm, mod_comm;
    MPI_Comm_split(comm, rank / Q, rank, &color_comm);
    MPI_Comm_split(comm, rank % Q, rank, &mod_comm);

    int color_rank, color_size, mod_rank, mod_size;
    MPI_Comm_rank(color_comm, &color_rank);
    MPI_Comm_size(color_comm, &color_size);
    MPI_Comm_rank(mod_comm, &mod_rank);
    MPI_Comm_size(mod_comm, &mod_size);

    cout << "Rank: " << rank << " Color Rank: " << color_rank << " Mod Rank: " << mod_rank << endl;

    MPI_Comm_free(&color_comm);
    MPI_Comm_free(&mod_comm);

    MPI_Finalize();
    return 0;
}