/*
Write an MPI program that builds a 2D process topology of shape P ×Q. On each column of processes, store a
vector x of length M , distributed in a linear load-balanced fashion “vertically” (it will be replicated Q times).
Start with data only in process (0,0), and scatter it down the first column. Once it is scattered on column 0,
broadcast it horizontally in each process row. Allocate a vector y of length M that is replicated “horizontally”
in each process row and stored also in linear load-balanced distribution; there will be P replicas, one in each
process row. Using MPI Allreduce or MPI Allgather with the appropriate communicators, do the parallel
copy y := x. There should be P replicas of the answer in y when you’re done.

Modification: Modify Problem #1 above, but store y in a scatter distribution (aka wrap-mapped) distribution. For this
case, from global coeffient J on Q processes, then local index is j = J/Q, q = J mod Q, and the number of
elements per process is the same as in the linear load-balanced distribution would produce with N elements
over Q partitions.
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
    int M = 25;
    vector<int> x(M);
    if (rank == 0){
        for (int i = 0; i < M; i++){
            x[i] = i + 1;
        }
    }

    int* sendcounts = new int[P];
    int* displs = new int[P];

    for (int i = 0; i < P; i++){
        sendcounts[i] = M / P;
        displs[i] = i * M / P;
    }

    // now, scatter it down the first column (make sure to use Scatterv, not Scatter)
    vector<int> x_local(M / P);
    MPI_Scatterv(x.data(), sendcounts, displs, MPI_INT, x_local.data(), M / P, MPI_INT, 0, color_comm);
    
    delete[] sendcounts;
    delete[] displs;

    // once it is scattered on column 0, broadcast it horizontally in each process row
    MPI_Bcast(x_local.data(), M / P, MPI_INT, 0, mod_comm);

    // allocate a vector y of length M that is replicated "horizontally" in each process row
    // there will be P replicas, one in each process row
    vector<int> y(M / P);

    // instead of using Allreduce or Allgather, we will use Scatterv to store y in a scatter distribution
    sendcounts = new int[Q];
    displs = new int[Q];

    for (int i = 0; i < Q; i++){
        sendcounts[i] = M / Q;
        displs[i] = i * M / Q;
    }

    // do the parallel copy y := x
    MPI_Scatterv(x_local.data(), sendcounts, displs, MPI_INT, y.data(), M / Q, MPI_INT, 0, mod_comm);

    delete[] sendcounts;
    delete[] displs;
    
    // print for debugging
    cout << "Rank " << rank << " has y = ";
    for (int i = 0; i < M / P; i++){
        cout << y[i] << " ";
    }
    cout << endl;
    
    MPI_Finalize();
    return 0;
}