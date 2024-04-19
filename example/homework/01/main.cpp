/*
Write an MPI program that builds a 2D process topology of shape P ×Q. On each column of processes, store a
vector x of length M , distributed in a linear load-balanced fashion “vertically” (it will be replicated Q times).
Start with data only in process (0,0), and scatter it down the first column. Once it is scattered on column 0,
broadcast it horizontally in each process row. Allocate a vector y of length M that is replicated “horizontally”
in each process row and stored also in linear load-balanced distribution; there will be P replicas, one in each
process row. Using MPI Allreduce or MPI Allgather with the appropriate communicators, do the parallel
copy y := x. There should be P replicas of the answer in y when you’re done.

Note utilize the code below.
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

    // here, we need to build a 2D process topology of shape P * Q
    // we already have the color_comm and mod_comm

    // we need to store a vector x of length M, distributed in a linear load-balanced fashion "vertically"
    // it will be replicated Q times
    int M = 10;
    vector<int> x(M);
    if (rank == 0){
        for (int i = 0; i < M; i++){
            x[i] = i;
        }
    }
    
    // scatter it down the first column
    vector<int> x_local(M / P);
    MPI_Scatter(&x[0], M / P, MPI_INT, &x_local[0], M / P, MPI_INT, 0, color_comm);
    
    // broadcast it horizontally in each process row
    vector<int> x_row(M / P);
    MPI_Bcast(&x_local[0], M / P, MPI_INT, 0, mod_comm);

    // allocate a vector y of length M that is replicated "horizontally" in each process row
    // there will be P replicas, one in each process row
    vector<int> y(M / P);

    // using MPI Allreduce or MPI Allgather with the appropriate communicators, do the parallel copy y := x
    MPI_Allreduce(&x_local[0], &y[0], M / P, MPI_INT, MPI_SUM, mod_comm);

    // there should be P replicas of the answer in y when you're done
    cout << "Rank " << rank << " y: ";
    for (int i = 0; i < M / P; i++){
        cout << y[i] << " ";
    }
    cout << endl;

    MPI_Comm_free(&color_comm);
    MPI_Comm_free(&mod_comm);

    MPI_Finalize();
    return 0;
}