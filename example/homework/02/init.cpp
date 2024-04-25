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

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);  
	Kokkos::initialize(argc, argv);
  	{
  	}
  	Kokkos::finalize();
	MPI_Finalize();
}
