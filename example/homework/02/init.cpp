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
		// Get rank and size
		MPI_Comm comm = MPI_COMM_WORLD;
		int rank, size;
		MPI_Comm_rank(comm, &rank);
		MPI_Comm_size(comm, &size);

		// Get number of rows, columns, and time steps
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
		}

		// Broadcast the number of rows, columns, and time steps
		MPI_Bcast(&nRows, 1, MPI_INT, 0, comm);
		MPI_Bcast(&nCols, 1, MPI_INT, 0, comm);
		MPI_Bcast(&nTime, 1, MPI_INT, 0, comm);

		// check if the number of rows is divisible by the number of processes
		int nRowsLocal = nRows / size;
		if (rank == (size-1)){
			nRowsLocal += nRows % size;
		}

		// the ghost cells are the cells that are on the edge of the domain
		int nRowsLocalWithGhost = nRowsLocal + 2;
		int nColsLocalWithGhost = nCols + 2;

		// initialize the domain
		Kokkos::View<int**> domain("domain", nRowsLocalWithGhost, nColsLocalWithGhost);
		Kokkos::View<int**> nextDomain("nextDomain", nRowsLocalWithGhost, nColsLocalWithGhost);

		int dims[2] = {0, 0};
		int sqrtSize = static_cast<int>(sqrt(size));
		if (sqrtSize * sqrtSize != size) {
			std::cout << "Number of processes must be a perfect square" << std::endl;
			MPI_Abort(comm, 1);
		}

		MPI_Dims_create(size, 2, dims);
		int periods[2] = {1, 1};
		MPI_Comm comm2D;
		MPI_Cart_create(comm, 2, dims, periods, 0, &comm2D);

		int rank2D;
		MPI_Comm_rank(comm2D, &rank2D);

		int north, south, east, west;
		MPI_Cart_shift(comm2D, 0, 1, &north, &south);
		MPI_Cart_shift(comm2D, 1, 1, &west, &east);

		int northwest, northeast, southwest, southeast;
		MPI_Cart_shift(comm2D, 0, 1, &northwest, &southeast);
		MPI_Cart_shift(comm2D, 1, 1, &southwest, &northeast);

		// fill the domain with random values
		Kokkos::View<int**> domainCopy("domainCopy", nRowsLocalWithGhost, nColsLocalWithGhost);
		srand(time(0));
		for (int i = 0; i < nRowsLocalWithGhost; i++) {
			for (int j = 0; j < nColsLocalWithGhost; j++) {
				domain(i, j) = rand() % 2;
			}
		}

		

		Kokkos::parallel_for(nRowsLocalWithGhost * nColsLocalWithGhost, KOKKOS_LAMBDA(const int i) {
			domain(i) = 0;
			domainCopy(i) = 0;
		});

		Kokkos::fence();

		srand(time(NULL));
		Kokkos::parallel_for(nRowsLocalWithGhost * nColsLocalWithGhost, KOKKOS_LAMBDA(const int i) {
			domain(i) = rand() % 2;
		});

		Kokkos::fence();


  	}
  	Kokkos::finalize();
	MPI_Finalize();
}
