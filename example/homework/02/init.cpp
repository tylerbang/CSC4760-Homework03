#include <Kokkos_Core.hpp>
#include <cstdio>

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);  
	Kokkos::initialize(argc, argv);
  	{

  	}
  	Kokkos::finalize();
	MPI_Finalize();
}
