using namespace std;
#include <iostream>
#include <assert.h>
#include <string>

#include <mpi.h>

#define GLOBAL_PRINT  // gather and print domains from process rank 0

// forward declarations:

class Process_2DGrid
{
public:
  Process_2DGrid(int _P, _Q, MPI_Comm _parent_comm) : the_P(_P), the_Q(_Q), the_parent_comm(_parent_comm) {MPI_Comm_size(_parent_comm,&parent_size);
                MPI_Comm_rank(_parent_comm,&myrank_in_parent_comm);
		pq_of_rank(myrank_in_parent, the_p, the_q);
		MPI_Comm_split(_parent_comm,the_p,myrank_in_parent_comm,&the_row_comm);
		MPI_Comm_split(_parent_comm,the_q,myrank_in_parent_comm,&the_col_comm);}
  // row_comm -- all processes in the same process row   participate [horizontal]
  // col_comm -- all processes in the same process column participate [vertical]
  
  virtual ~Process_2DGrid() {MPI_Comm_free(&the_row_comm);
                             MPI_Comm_free(&the_col_comm);}

  int P() const {return the_P;}
  int Q() const {return the_Q;}
  int p() const {return the_p;}
  int q() const {return the_q;}

  int  rank_of_pq(int _p, int _q) const {return _p*Q+_q;}
  void pq_of_rank(int R, int &_p, int &_q) const {_p=R/Q;_q=R%Q;}
  
  MPI_Comm parent_comm() const {return the_parent_comm;}
  MPI_Comm row_comm()    const {return the_row_comm;}
  MPI_Comm col_comm()    const {return the_col_comm;}
  
protected:
  int the_P, the_Q; // shape of grid
  int the_p, the_q; // this process' location in the grid
  int parent_size;
  int myrank_in_parent_comm;
  MPI_Comm the_parent_comm;
  MPI_Comm the_row_comm;
  MPI_Comm the_col_comm;
};

class LinearDistribution
{
public:

  LinearDistribution(int _P, _M) : the_P(_P), the_M(_M) {nominal=the_M/the_P;
                                      extra=the_M%the_P; factor1=extra*(nominal+1);}
  void global_to_local(int I, int &p, int &i) const
              {p = (I < factor1) ? I/(nominal+1) : extra+((I-factor1)/nominal);
	       i = I - (p < extra) ? p*(nominal+1) : factor1+(p-extra)*nominal;}
  int local_to_global(int p, int i) const
       {return i +     (p < extra) ? p*(nominal+1) : factor1+(p-extra)*nominal;}

  int m(int p) const {return (p < extra) ? (nominal+1) : nominal;}
  
  int M() const {return the_M;}
  int P() const {return the_P;}

protected:
  int the_M, the_P, nominal, extra, factor1;
};
  
class Domain // this will be integrated with a Kokkos view and subview in HW#3.
{
public:
  Domain(int _M, int _N, int _halo_depth, const char *_name="") :
        exterior(new char[(_M+2*_halo_depth)*(_N+2*_halo_depth)]), M(_M), N(_N),
	halo_depth(_halo_depth), name(_name)
        {interior = exterior+(N+2*halo_depth)*halo_depth + halo_depth;}
  
  virtual ~Domain() {delete[] exterior;}
  
  char &operator()(int i, int j)       {return interior[i*(N+2*halo_depth)+j];}
  char operator() (int i, int j) const {return interior[i*(N+2*halo_depth)+j];}

  int rows() const {return M;}
  int cols() const {return N;} // actual domain size.

  const string &myname() const {return name;}

  char *rawptr()    {return exterior;}
  char *cookedptr() {return interior;}  

protected:
  char *exterior, *interior;
  int M;
  int N;
  int halo_depth;

  string name;
};

// forward declarations:
void zero_domain(Domain &domain); // zeros the halos too.
void print_domain(Domain &domain, int p, int q); // prints only the true domain
void update_domain(Domain &new_domain, Domain &old_domain, Process_2DGrid &grid);
void parallel_code(int M, int N, int iterations, Process_2DGrid &grid);

int main(int argc, char **argv)
{
  // command-line-specified parameters:
  int M, N;
  int P, Q;
  int iterations;

  if(argc < 6)
  {
    cout << "usage: " << argv[0] << " M N P Q iterations" << endl;
    exit(0);
  }

  int size, myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  if(P*Q != size)
  {
    cout << "grid mismatch: " << argv[0] << P << "*" << Q << "must be size of world: "<< size <<"!" << endl; 
    exit(0);
  }
  
  int array[5];
  if(0 == myrank)
  {
     M = atoi(argv[1]); N = atoi(argv[2]);
     P = atoi(argv[3]); Q = atoi(argv[4]);
     iterations = atoi(argv[5]);

     array[0] = M;
     array[1] = N;
     array[2] = P;
     array[3] = Q;
     array[4] = iterations;
  }
  MPI_Bcast(array, 5, MPI_INT, 0, MPI_COMM_WORLD);
  if(myrank != 0)
  {
    M = array[0];
    N = array[1];
    P = array[2];
    Q = array[3];
    iterations = array[4];
  }

  Process_2DGrid grid(P, Q, MPI_COMM_WORLD);
  parallel_code(M, N, iterations, grid);
  
  MPI_Finalize();
}

void parallel_code(int M, int N, int iterations, Process_2DGrid &grid)
{
  // use the linear, load-balanced distribution in both dimensions:
  LinearDistribution row_distribution(M, grid.P());
  LinearDistribution col_distribution(N, grid.Q());

  // mxn is the size of the domain (p,q):
  int m = row_distribution.m(grid.p());
  int n = col_distribution.m(grid.q());

  // local domains (replace with Kokkos View and Subview in Kokkos version!):
  Domain even_domain(m,n,1,"even local domain"); // halo depths of 1
  Domain odd_domain (m,n,1,"odd local domain");  // mxn size, varies per p,q location

  zero_domain(even_domain);
  zero_domain(odd_domain);

#ifdef GLOBAL_PRINT // not scalable in memory or I/O to print each iteration, for testing.
  Domain *global_domain = nullptr;
  Domain *domain_slice = nullptr;
  int *rowcounts = nullptr;
  int *rowdispls = nullptr;

  if(grid.q() == 0) // left column of grid only does this distribution.
  {
    domain_slice = new Domain(m,N,0,"slice of domain"); // no halos, 1D decomp.
                                                        // mxN size
    
    if(grid.p() == 0) // root of first [vertical] scatter
    {
      rowcounts = new int[grid.P()];
      rowdispls = new int[grid.P()];

      for(int phat = 0; phat < grid.P(); ++phat)
	rowcounts[phat] = row_distribution.m(phat)*col_distribution.M();  //whole cols.

      int count = displs[0] = 0;
      for(int phat = 1; phat < grid.P(); ++phat)
      {
        count += rowcounts[phat-1];
        rowdispls[phat] = count;
      }

      // process (0,0) holds an initial copy of the whole domain [to illustrate
      // scatter, to help with I/O.  you don't do this in really scalable programs.]
      global_domain = new Domain(M,N,0,"Global Domain"); // no halos, MxN size
      zero_domain(*global_domain);

      if((N >= 8) && (M >= 10))
      {
	// blinker at top left, touching right...
	(*global_domain)(0,(N-1)) = 1;
	(*global_domain)(0,0)     = 1;
	(*global_domain)(0,1)     = 1;

	// and a glider:
	(*global_domain)(8,5)     = 1;
	(*global_domain)(8,6)     = 1;
	(*global_domain)(8,7)     = 1;
	(*global_domain)(7,7)     = 1;
	(*global_domain)(6,6)     = 1;
      }

    } // if(grid.my_p() == 0

    MPI_Scatterv((grid.p()==0) ? global_domain->rawptr() : nullptr,
		 rowcounts, rowdispls, MPI_CHAR, domain_slice->rawptr(),
                 m*N, MPI_CHAR, 0 /* root */, grid.col_comm() /*vertical split */);

    //depending on if print uses these, otherwise could delete here.
    //if(grid.p() == 0)
    //{  
    //delete[] rowcounts;
    //delete[] rowdispls;
    //}

  } // end if(grid.my_q() == 0)

  int *colcounts = nullptr;
  int *coldispls = nullptr;

  // process column zero has vertically scattered domain in 1D so far.
  // all process rows execute the next if(grid.q() == 0) ... independently, in parallel:
  if(grid.q() == 0) // now work to scatter horizontally in each process row.
  {
      // domain slice in leftmost column must be distributed into even local domain.
      colcounts = new int[grid.Q()];
      coldispls = new int[grid.Q()];

      for(int qhat = 0; qhat < grid.Q(); ++qhat)
	thecolcounts[qhat] = col_distribution.m(qhat);  /* notice it is just col count*/

      int count = displs[0] = 0;
      for(int qhat = 1; qhat < grid.Q(); ++qhat)
      {
        count += m*thecolcounts[qhat-1]; // but this accounts for total area per process
        coldispls[qhat] = count;
      }
  }
  
  // processes with grid.q()==0 use this:
  MPI_Type vertical_slice_in;  // notice global stride of N
  if(grid.q()==0)
  {
     MPI_Type_vector(m /* count */, 1 /* blocklen */, N /* stride, no halos */,
		     MPI_CHAR, &vertical_slice_in);
     MPI_Type_commit(&vertical_slice_in);
  }

  // all processes need this:
  MPI_Type vertical_slice_out; // notice local stride of n+2
  int MPI_Type_vector(m /* count */, 1 /* blocklen */, n+2*1 /* stride incl halos(1) */,
		      MPI_CHAR, &vertical_slice_out);
  MPI_Type_commit(&vertical_slice_out);

  // we point at the interior of the even_domain as the base for transfer:
  MPI_Scatterv((grid.q()==0) ? domain_slice->rawptr() : nullptr,
	       colcounts, coldispls, vertical_slice_in, even_domain->cookedptr, n,
	       vertical_slice_out, 0 /* root */, grid.row_comm());

  //move down: MPI_Type_free(&vertical_slice_in);
  //move down: MPI_Type_free(&vertical_slice_out);
  //move down:
  //  if(grid.q() == 0)
  //{
  //delete[] thecolcounts;
  //delete[] coldispls;
  //  }
  
#else  // local domain test fills.
  // fill in even_domain with something meaningful (initial state)
  // this requires min size for default values to fit:
  if((n >= 8) && (m >= 10))
  {
#if 0    
    even_domain(0,(n-1)) = 1;
    even_domain(0,0)     = 1;
    even_domain(0,1)     = 1;
    
    even_domain(3,5) = 1;
    even_domain(3,6) = 1;
    even_domain(3,7) = 1;

    even_domain(6,7) = 1;
    even_domain(7,7) = 1;
    even_domain(8,7) = 1;
    even_domain(9,7) = 1;
#else
    // blinker at top left, touching right...
    even_domain(0,(n-1)) = 1;
    even_domain(0,0)     = 1;
    even_domain(0,1)     = 1;

    // and a glider:
    even_domain(8,5)     = 1;
    even_domain(8,6)     = 1;
    even_domain(8,7)     = 1;
    even_domain(7,7)     = 1;
    even_domain(6,6)     = 1;
#endif    
  }
#endif  

#ifdef GLOBAL_PRINT
    if(grid.p()==0 && grid.q()==0)
    {  
      cout << "Initial State:" << endl;
      print_domain(*global_domain, 0,0);
    }
#else
    cout << "Initial State:" << i << endl;
    print_domain(*even, p, q);
#endif  

  Domain *odd, *even; // pointer swap magic
  odd = &odd_domain;
  even = &even_domain;

  for(int i = 0; i < iterations; ++i)
  {
    update_domain(*odd, *even, grid);

#ifdef GLOBAL_PRINT
    if(grid.p()==0 && grid.q()==0)
    {
      // gather the domains from all PxQ processes back into global_domain,
      // but not their halos... the data to print is in the odd ptr.
      
      print_domain(*global_domain, 0, 0);
    }
#else
    cout << "Iteration #" << i << endl; print_domain(*odd, p, q);
#endif

    // swap pointers:
    Domain *temp = odd;
    odd  = even;
    even = temp;
  }
  
#ifdef GLOBAL_PRINT
  if(grid.p() == 0 && grid.q() == 0)
  {
    delete global_domain;
  }

  if(grid.q() == 0) // free memory & datatypes associated with first scatter/gather
  {
    delete[] domain_slice;
    delete[] colcounts;
    delete[] coldispls;
  }

  if(grid.p() == 0) // free memory & datatypes associated with the second scatter/gather
  {
    delete[] rowcounts;
    delete[] rowdispls;
  }
  
  // free any other memory & datatypes applying to all (p,q)...
  
#endif  
}

void zero_domain(Domain &domain)
{
  for(int i = 0; i < domain.rows(); ++i)
    for(int j = 0; j < domain.cols(); ++j)
      domain(i,j) = 0;
}

void print_domain(Domain &domain, int p, int q)
{
  cout << "(" << p << "," << q << "): " << domain.myname() << ":" <<endl;
  // this is naive; it doesn't understand big domains at all 
  for(int i = 0; i < domain.rows(); ++i)
  {
    for(int j = 0; j < domain.cols(); ++j)
      cout << (domain(i,j) ? "*" : ".");
    cout << endl;
  }
}

inline char update_the_cell(char cell, int neighbor_count) //Life cellular a rule.
{
  char newcell;
#if 1
  if(cell == 0) // dead now
    newcell = (neighbor_count == 3) ? 1 : 0;
  else // was live, what about now?
    newcell = ((neighbor_count == 2)||(neighbor_count == 3)) ? 1 : 0;
#else
  // alt logic: notice the cell is always live next generation, if it has three neighbors
  if(neighbor_count == 3)
    newcell = 1;
  else if((cell==1)&&(neighbor_count == 2)) newcell = 1;
  else
    newcell = 0;
  return newcell;
}
      
void update_domain(Domain &new_domain, Domain &old_domain, Process_2DGrid &grid)
{
  MPI_Request request[16];
  
  int m = new_domain.rows();
  int n = new_domain.cols();

  // simplest (but slower) to use buffers to hold halos (all contiguous):
  char *top_row    = new char[n]; char *top_halo    = new char[n];
  char *bottom_row = new char[n]; char *bottom_halo = new char[n];
  char *left_col   = new char[m]; char *left_halo   = new char[n];
  char *right_col  = new char[m]; char *right_halo  = new char[n]; 
  char NW_halo, NE_halo, SW_halo, SE_halo; // don't need to copy corners to send

  const int TOP_HALO = 0, BOTTOM_HALO = 1, WEST_HALO = 2, EAST_HALO = 3,
            NW_HALO  = 4, NE_HALO     = 5, SW_HALO   = 6, SE_HALO   = 7;

  int North_prank = (grid.p()-1+grid.P())%grid.P();
  int South_prank = (grid.p()+1         )%grid.P();
  int West_qrank  = (grid.q()-1+grid.Q())%grid.Q();
  int East_qrank  = (grid.q()+1         )%grid.Q();
  int NW_rank     = grid.rank_of_pq((grid.p()-1+grid.P())%grid.P(),
				    (grid.q()-1+grid.Q())%grid.Q());
  int NE_rank     = grid.rank_of_pq((grid.p()-1+grid.P())%grid.P(),
				    (grid.q()+1+grid.Q())%grid.Q());
  int SW_rank     = grid.rank_of_pq((grid.p()+1+grid.P())%grid.P(),
				    (grid.q()-1+grid.Q())%grid.Q());
  int SE_rank     = grid.rank_of_pq((grid.p()+1+grid.P())%grid.P(),
				    (grid.q()+1+grid.Q())%grid.Q());
  
  // use the col_comm and row_comm communicators for 4 cardinal direction:
  MPI_Irecv(top_halo,    n, MPI_CHAR, North_prank, TOP_HALO,    grid.col_comm(), &request[0]);
  MPI_Irecv(bottom_halo, n, MPI_CHAR, South_prank, BOTTOM_HALO, grid.col_comm(), &request[1]);
  MPI_Irecv(left_halo,   m, MPI_CHAR, West_qrank,  LEFT_HALO,   grid.row_comm(), &request[2]);
  MPI_Irecv(right_halo,  m, MPI_CHAR, East_qrank,  RIGHT_HALO,  grid.row_comm(), &request[3]);

  MPI_Irecv(&NW_halo, 1, MPI_CHAR, NW_rank, NW_HALO, grid.parent_comm(), &request[4]);
  MPI_Irecv(&NE_halo, 1, MPI_CHAR, NE_rank, NE_HALO, grid.parent_comm(), &request[5]);
  MPI_Irecv(&SW_halo, 1, MPI_CHAR, SW_rank, SW_HALO, grid.parent_comm(), &request[6]);
  MPI_Irecv(&SE_halo, 1, MPI_CHAR, SE_rank, SE_HALO, grid.parent_comm(), &request[7]);
  
  // 2. gather+send my top row and bottom row to adjacent process
  for(int j = 0; j < n; ++j)   // fill the top row
  {
      top_row[j] = old_domain(top_row_index,j);
  }
  MPI_Isend(top_row, n, MPI_CHAR, North_prank, BOTTOM_HALO, grid.col_comm(), &request[8]);

  for(int j = 0; j < n; ++j) // fill in the bottom row
  {
     bottom_row[j] = old_domain(bottom_row_index,j);
  }
  MPI_Isend(bottom_row, n, MPI_CHAR, South_prank, TOP_HALO, grid.col_comm(), &request[9]);

  // ADD: rest of sends...
  
  // complete all 16 transfers
  MPI_Waitall(16, request, MPI_STATUSES_IGNORE);

  // the entire perimeter must now be copied into old_domain's halo space.
  // ADD.

  // the entirety of the domain is computed so {replace with Kokkos kernel for HW#3}
  for(int i = 0; i < m ; ++i)
    for(int j = 0; j < n; ++j)
    {
      int neighbor_count =
         old_domain(i-1,j-1)+old_domain(i-1,j)+old_domain(i-1,j+1)
	+old_domain(i,  j-1)+0                +old_domain(i,  j+1)
	+old_domain(i+1,j-1)+old_domain(i+1,j)+old_domain(i+1,j+1);
      new_domain(i,j) = update_the_cell(old_domain(i,j), neighbor_count);
    } // int j,i

  // remember, in a performant code, we would encapsulate the
  // dynamic memory allocation once level higher in the code...
  delete[] top_row, top_halo;
  delete[] bottom_row, bottom_halo;
  delete[] left_col, left_halo;
  delete[] right_col, right_halo;
}


