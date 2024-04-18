#include <Kokkos_Core.hpp>
#include <iostream>

template <typename View_t>
typename View_t::non_const_value_type neighbor_reduce(View_t subv, int x, int y){
  typename View_t::non_const_value_type accer = 0;
  for(int i = x-1; i<=(x+1); i++){
    for(int j = y-1; j<=(y+1); j++){
      accer += subv(i,j);
    }
  }
  return accer;
}

int main(int argc, char** argv) {
  int error = 0;

  Kokkos::initialize( argc, argv );
  {
    int n = 10;
    Kokkos::View<int**> A("A",n, n);
    Kokkos::View<int**> B("B",n, n);

    Kokkos::parallel_for(A.extent(0), KOKKOS_LAMBDA(int i){
        Kokkos::parallel_for(A.extent(1), KOKKOS_LAMBDA(int j){//for(int j = 0; j<A.extent(1); j++){
          A(i,j) = i+1*(j+1);
          std::cout << A(i, j) << " ";
          });
        std::cout << std::endl;
      });
    Kokkos::fence();
    std::cout << "\n\n\n";
    for(int i = 1; i<B.extent(0)-1; i++){
        for(int j = 1; j<A.extent(1)-1; j++){
          B(i,j) = neighbor_reduce(A, i, j);
          }
      }
    Kokkos::parallel_for(B.extent(0), KOKKOS_LAMBDA(int i){
        for(int j = 0; j<B.extent(1); j++){
          std::cout << B(i, j) << " ";
          }
        std::cout << std::endl;
      });
  }
  Kokkos::finalize();

  return error;
}
