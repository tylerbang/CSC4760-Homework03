/*
Modify your Game of Life program from Homework #2, Problem #2 to use a Kokkos parallel kernel to
compute the updates at each iteration.
Hints: you may wish to change your local domain data structures to include the halos (ghost cells) as
part of your local domains. This will remove the need for ’if’ in the Kokkos kernel. At iteration i, your
receives of halos will be into the domain to be updated. This will mean you have to have two sets of halos,
one in each of the even and odd copies of your domain.
Note: Please look at this problem soon, so that you can ask questions on 4/18/24 and 4/23/24 in class
to be sure you get any needed questions answered. Class mentor Evan Suggs will be available to advise on
Kokkos issues too.
*/

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cstdio>
#include <bits/stdc++.h>
#include <cstdlib>
#include <ctime>

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	Kokkos::initialize(argc, argv);
  	{
		MPI_Comm comm = MPI_COMM_WORLD;
		int rank, size;
		MPI_Comm_rank(comm, &rank);
		MPI_Comm_size(comm, &size);

		// variables for the number of rows, columns, and time steps
		int nRows;
		int nCols;
		int nTime;

		// Check the number of arguments
		if (rank == 0) {
			if (argc != 4) {
				std::cout << "Usage: mpirun -n <number of processes> ./init <nRows> <nCols> <nTime>" << std::endl;
				MPI_Abort(comm, 1);
			}
			nRows = atoi(argv[1]);
			nCols = atoi(argv[2]);
			nTime = atoi(argv[3]);
			if (nRows < 1 || nCols < 1 || nTime < 1) {
				std::cout << "nRows, nCols, nTime must be greater than 0" << std::endl;
				MPI_Abort(comm, 1);
			}
			int sqrtSize = static_cast<int>(std::sqrt(size));
			if (sqrtSize * sqrtSize != size) {
				std::cout << "Number of processes must be a perfect square" << std::endl;
				MPI_Abort(comm, 1);
			}	
		}

		// Broadcast the number of rows, columns, and time steps
		MPI_Bcast(&nRows, 1, MPI_INT, 0, comm);
		MPI_Bcast(&nCols, 1, MPI_INT, 0, comm);
		MPI_Bcast(&nTime, 1, MPI_INT, 0, comm);

		// Check if the number of rows is divisible by the number of processes
		int nRowsLocal = nRows / size;
		int remainder = nRows % size;

		if (rank < remainder) {
			nRowsLocal++;
		}

		// the ghost cells are the cells that are on the edge of the domain
		int nRowsLocalWithGhost = nRowsLocal + 2;
		int nColsWithGhost = nCols + 2;

		// Initialize the domain (remember, we are using Kokkos views)
		Kokkos::View<int**> currDomain("currDomain", nRowsLocal + 2, nCols + 2);
		Kokkos::View<int**> nextDomain("nextDomain", nRowsLocal + 2, nCols + 2);

		// fill the domain with random values
		srand(time(0) + rank);
		for (int iRow = 1; iRow <= nRowsLocal; iRow++) {
			for (int iCol = 1; iCol <= nCols; iCol++) {
				currDomain(iRow, iCol) = rand() % 2;
			}
		}

		int* sendcounts = new int[size];
		int* displs = new int[size];

		for (int i = 0; i < size; i++) {
    		sendcounts[i] = (nRows / size + (i < remainder ? 1 : 0)) * nCols;
    		displs[i] = (i * nRows / size + std::min(i, remainder)) * nCols;
		}
		if (rank == 0) {
    		MPI_Scatterv(&currDomain(1, 1), sendcounts, displs, MPI_INT, MPI_IN_PLACE, 0, MPI_INT, 0, comm);
		} else {
    		MPI_Scatterv(NULL, NULL, NULL, MPI_INT, &currDomain(1, 1), nRowsLocal * nCols, MPI_INT, 0, comm);
		}
		delete[] sendcounts;
		delete[] displs;

		int dims[2] = {0, 0};
		MPI_Dims_create(size, 2, dims);
		int periods[2] = {1, 1};
		MPI_Comm comm2D;
		MPI_Cart_create(comm, 2, dims, periods, 0, &comm2D);

		// get the rank in the new communicator
		int rank2D;
		MPI_Comm_rank(comm2D, &rank2D);

		// calculate the von Neumann neighbors
		int north, south, east, west;
		MPI_Cart_shift(comm2D, 0, 1, &north, &south);
		MPI_Cart_shift(comm2D, 1, 1, &west, &east);

		// calculate the Moore neighbors
		int northwest, northeast, southwest, southeast;
		MPI_Cart_shift(comm2D, 0, 1, &northwest, &southeast);
		MPI_Cart_shift(comm2D, 1, 1, &southwest, &northeast);
		MPI_Cart_shift(comm2D, 0, 1, &northwest, &northeast);
		MPI_Cart_shift(comm2D, 1, 1, &southwest, &southeast);

		const int ALIVE = 1;
		const int DEAD = 0;

		// vscode is a slow piece of garbage
		// I want to use emacs
		// but sadly I cannot because there
		// is no copilot for emacs
		// sadge

		// Brock Purdy is the greatest Mr. Irrelevant of all time

		// now we can start the simulation
		for (int iTime = 0; iTime < nTime; iTime++) {

				// convert to separate sends and receives
				MPI_Send(&currDomain(1, 1), nCols, MPI_INT, north, 0, comm2D);
				MPI_Send(&currDomain(nRowsLocal, 1), nCols, MPI_INT, south, 0, comm2D);
				MPI_Send(&currDomain(1, 1), 1, MPI_INT, west, 0, comm2D);
				MPI_Send(&currDomain(1, nCols), 1, MPI_INT, east, 0, comm2D);
				MPI_Send(&currDomain(1, 1), 1, MPI_INT, northwest, 0, comm2D);
				MPI_Send(&currDomain(1, nCols), 1, MPI_INT, northeast, 0, comm2D);
				MPI_Send(&currDomain(nRowsLocal, 1), 1, MPI_INT, southwest, 0, comm2D);
				MPI_Send(&currDomain(nRowsLocal, nCols), 1, MPI_INT, southeast, 0, comm2D);

				MPI_Recv(&currDomain(nRowsLocal + 1, 1), nCols, MPI_INT, south, 0, comm2D, MPI_STATUS_IGNORE);
				MPI_Recv(&currDomain(0, 1), nCols, MPI_INT, north, 0, comm2D, MPI_STATUS_IGNORE);
				MPI_Recv(&currDomain(1, nCols + 1), 1, MPI_INT, east, 0, comm2D, MPI_STATUS_IGNORE);
				MPI_Recv(&currDomain(1, 0), 1, MPI_INT, west, 0, comm2D, MPI_STATUS_IGNORE);
				MPI_Recv(&currDomain(nRowsLocal + 1, nCols + 1), 1, MPI_INT, southeast, 0, comm2D, MPI_STATUS_IGNORE);
				MPI_Recv(&currDomain(nRowsLocal + 1, 0), 1, MPI_INT, southwest, 0, comm2D, MPI_STATUS_IGNORE);
				MPI_Recv(&currDomain(0, nCols + 1), 1, MPI_INT, northeast, 0, comm2D, MPI_STATUS_IGNORE);
				MPI_Recv(&currDomain(0, 0), 1, MPI_INT, northwest, 0, comm2D, MPI_STATUS_IGNORE);

			// updoot domainy boi
			Kokkos::parallel_for("updateDomain", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {nRowsLocalWithGhost, nColsWithGhost}), KOKKOS_LAMBDA (const int iRow, const int iCol) {
				int nNeighbors =
					(iRow > 0 && iCol > 0 ? currDomain(iRow-1, iCol-1) : 0) +
					(iRow > 0 ? currDomain(iRow-1, iCol) : 0) +
					(iRow > 0 && iCol < nColsWithGhost-2 ? currDomain(iRow-1, iCol+1) : 0) +
					(iCol > 0 ? currDomain(iRow, iCol-1) : 0) +
					(iCol < nColsWithGhost-2 ? currDomain(iRow, iCol+1) : 0) +
					(iRow < nRowsLocalWithGhost-2 && iCol > 0 ? currDomain(iRow+1, iCol-1) : 0) +
					(iRow < nRowsLocalWithGhost-2 ? currDomain(iRow+1, iCol) : 0) +
					(iRow < nRowsLocalWithGhost-2 && iCol < nColsWithGhost-2 ? currDomain(iRow+1, iCol+1) : 0);
				if (currDomain(iRow, iCol) == ALIVE) {
					if (nNeighbors < 2 || nNeighbors > 3) {
						nextDomain(iRow, iCol) = 0;
					} else {
						nextDomain(iRow, iCol) = 1;
					}
				} else {
					if (nNeighbors == 3) {
						nextDomain(iRow, iCol) = 1;
					} else {
						nextDomain(iRow, iCol) = 0;
					}
				}
			});
			Kokkos::fence();

			// copy the next domain to the current domain
			Kokkos::deep_copy(currDomain, nextDomain);
			Kokkos::fence();

			// print each iteration
			if (rank == 0) {
				std::cout << "Iteration " << iTime << std::endl;
				std::cout << "--------------------------------" << std::endl;
				for (int iRow = 1; iRow <= nRows; iRow++) {
					for (int iCol = 1; iCol <= nCols; iCol++) {
						std::cout << currDomain(iRow, iCol) << " ";
					}
					std::cout << std::endl;
				}
			}
		}
  	}
  	Kokkos::finalize();
	MPI_Finalize();
}

// I hate M