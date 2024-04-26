using namespace std;
#include <mpi.h>
#include <iostream>
#include <bits/stdc++.h>
#include <cstdlib>
#include <ctime>

// this only works for a world that is greater than 8x8, I know you said 8x8, but I'm just gonna do 10x10 because I'm lazy
// this is main function
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    // get that rank and size
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // variables for the number of rows, columns, and time steps
    int nRows;
    int nCols;
    int nTime;

    // let's just say that you run it like this:
    // mpirun -np 4 ./init [nRows] [nCols] [nTime (or just iterations, you get the point)]
    // so we need to check if the number of arguments is correct
    // if it's not, we just abort the program

    // preferably, you want to run this on a heckin cluster with them sweet sweet nodes

    // also, I know you said to use * for alive and space for dead, but I'm just gonna use 1 and 0 because I'm lazy

    // Check the number of arguments
    if (rank == 0) {
        if (argc != 4) {
            cout << "Usage: mpirun -n <number of processes> ./init <nRows> <nCols> <nTime>" << endl;
            MPI_Abort(comm, 1);
        }
        nRows = atoi(argv[1]);
        nCols = atoi(argv[2]);
        nTime = atoi(argv[3]);
        if (nRows < 1 || nCols < 1 || nTime < 1) {
            cout << "nRows, nCols, nTime must be greater than 0" << endl;
            MPI_Abort(comm, 1);
        }
    }

    // Broadcast the number of rows, columns, and time steps
    MPI_Bcast(&nRows, 1, MPI_INT, 0, comm);
    MPI_Bcast(&nCols, 1, MPI_INT, 0, comm);
    MPI_Bcast(&nTime, 1, MPI_INT, 0, comm);

    // Check if the number of rows is divisible by the number of processes
    int nRowsLocal = nRows / size;
    if (rank == (size-1)){
       nRowsLocal += nRows % size; 
    }

    // the ghost cells are the cells that are on the edge of the domain
    int nRowsLocalWithGhost = nRowsLocal + 2;
    int nColsWithGhost = nCols + 2;

    // Initialize the domain
    vector<vector<int>> currDomain(nRowsLocalWithGhost, vector<int>(nColsWithGhost, 0));
    vector<vector<int>> nextDomain(nRowsLocalWithGhost, vector<int>(nColsWithGhost, 0));

    int dims[2] = {0, 0};
    int sqrtSize = static_cast<int>(sqrt(size));
    if (sqrtSize * sqrtSize != size){
        cout << "The number of processes must be a perfect square" << endl;
        MPI_Abort(comm, 1);
    }

    MPI_Dims_create(size, 2, dims);
    int periods[2] = {1, 1};
    MPI_Comm comm2D;
    MPI_Cart_create(comm, 2, dims, periods, 0, &comm2D);

    //get the rank in the new communicator
    int rank2D;
    MPI_Comm_rank(comm2D, &rank2D);

    //calculate the neighbors
    int north, south, east, west;
    MPI_Cart_shift(comm2D, 0, 1, &north, &south);
    MPI_Cart_shift(comm2D, 1, 1, &west, &east);

    // for diagonal neighbors
    int northwest, northeast, southwest, southeast;
    MPI_Cart_shift(comm2D, 0, 1, &northwest, &southeast);
    MPI_Cart_shift(comm2D, 1, 1, &southwest, &northeast);
   
    // fill the domain with random values
    srand(time(0));
    for (int iRow = 1; iRow <= nRowsLocal; iRow++){
        for (int iCol = 1; iCol <= nColsWithGhost; iCol++){
            currDomain[iRow][iCol] = rand() % 2;
        }
    }
    
    const int ALIVE = 1;
    const int DEAD = 0;
    // NOTE: The main loop is not correct, you need to modify it to work for a 2D domain, but first you need to split the communication part into two parts
    // otherwise, you will get errors
    // Main loop this only works for a 1D domain, so we need to modify it to work for a 2D domain
    for (int iTime = 0; iTime < nTime; iTime++){

       // send to north and receive from south
        MPI_Send(&currDomain[1][1], nCols, MPI_INT, north, 0, comm);
        MPI_Recv(&currDomain[nRowsLocal + 1][1], nCols, MPI_INT, south, 0, comm, MPI_STATUS_IGNORE);

        // send to south and receive from north
        MPI_Send(&currDomain[nRowsLocal][1], nCols, MPI_INT, south, 0, comm);
        MPI_Recv(&currDomain[0][1], nCols, MPI_INT, north, 0, comm, MPI_STATUS_IGNORE);

        // send to west and receive from east
        MPI_Send(&currDomain[1][1], 1, MPI_INT, west, 0, comm);
        MPI_Recv(&currDomain[1][nCols + 1], 1, MPI_INT, east, 0, comm, MPI_STATUS_IGNORE);

        // send to east and receive from west
        MPI_Send(&currDomain[1][nCols], 1, MPI_INT, east, 0, comm);
        MPI_Recv(&currDomain[1][0], 1, MPI_INT, west, 0, comm, MPI_STATUS_IGNORE);

        // send to northwest and receive from southeast
        MPI_Send(&currDomain[1][1], 1, MPI_INT, northwest, 0, comm);
        MPI_Recv(&currDomain[nRowsLocal + 1][nCols + 1], 1, MPI_INT, southeast, 0, comm, MPI_STATUS_IGNORE);

        // send to northeast and receive from southwest
        MPI_Send(&currDomain[1][nCols], 1, MPI_INT, northeast, 0, comm);
        MPI_Recv(&currDomain[nRowsLocal + 1][0], 1, MPI_INT, southwest, 0, comm, MPI_STATUS_IGNORE);

        // send to southwest and receive from northeast
        MPI_Send(&currDomain[nRowsLocal][1], 1, MPI_INT, southwest, 0, comm);
        MPI_Recv(&currDomain[0][nCols + 1], 1, MPI_INT, northeast, 0, comm, MPI_STATUS_IGNORE);

        // send to southeast and receive from northwest
        MPI_Send(&currDomain[nRowsLocal][nCols], 1, MPI_INT, southeast, 0, comm);
        MPI_Recv(&currDomain[0][0], 1, MPI_INT, northwest, 0, comm, MPI_STATUS_IGNORE);

        // Update the next domain
        for (int iRow = 0; iRow < nRowsLocalWithGhost; iRow++){
            currDomain[iRow][0] = currDomain[iRow][nCols];
            currDomain[iRow][nCols + 1] = currDomain[iRow][1];
        }

        // if not the first process, receive token from previous process
        if (rank != 0){
            for (int iRow = 1; iRow <= nRowsLocal; iRow++){
                MPI_Send(&currDomain[iRow][1], nCols, MPI_INT, 0, 0, comm);
            }
        }

        // if the first process, receive token from last process
        if (rank == 0){
            cout << "iTime: " << iTime << endl;

            for (int iRow = 1; iRow <= nRowsLocal; iRow++){
                for (int iCol = 1; iCol <= nCols; iCol++){
                    cout << currDomain[iRow][iCol] << " ";
                }
                cout << endl;
            }
            for (int sourceRank = 1; sourceRank < size; sourceRank++){
                int nRecv = nRows / size;
                if (sourceRank == size - 1){
                    nRecv += nRows % size;
                }
                vector<int> buff(nCols, 0);
                for (int iRecv = 0; iRecv < nRecv; iRecv++){
                    MPI_Recv(&buff[0], nCols, MPI_INT, sourceRank, 0, comm, MPI_STATUS_IGNORE);
                    for (int i : buff){
                        cout << i << " ";
                    }
                    cout << endl;
                }
            }
        }
        
                // Update the next domain
        for (int iRow = 1; iRow <= nRowsLocal; iRow++){
            for (int iCol = 1; iCol <= nCols; iCol++){
                int nAliveNeighbors = 0;
                for (int jRow = max(1, iRow - 1); jRow <= min(nRowsLocal, iRow + 1); jRow++){
                    for (int jCol = max(1, iCol - 1); jCol <= min(nCols, iCol + 1); jCol++){
                        if (((jRow != iRow) || (jCol != iCol)) && currDomain[jRow][jCol] == ALIVE){
                            nAliveNeighbors++;
                        }
                    }
                }

                if (currDomain[iRow][iCol] == ALIVE){
                   nextDomain[iRow][iCol] = (nAliveNeighbors == 2 || nAliveNeighbors == 3) ? ALIVE : DEAD;
                } else {
                     nextDomain[iRow][iCol] = (nAliveNeighbors == 3) ? ALIVE : DEAD;
                }
            }
        }

        for (int iRow = 1; iRow <= nRowsLocal; iRow++){
            for (int iCol = 1; iCol <= nCols; iCol++){
                currDomain[iRow][iCol] = nextDomain[iRow][iCol];
            }
        }
    }
    MPI_Finalize();    
    return 0;
}
