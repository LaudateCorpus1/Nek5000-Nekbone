# NEKbone code with one-sided refactorization.
There are three versions of code with MPI one-sided calls using different synchronization schemes: 
- Baseline code: Directory "src" has baseline code with two-sided MPI routines.
- Global synchronization: Directory "src_RMA_GS" has code with MPI one-sided implementation using global synchronization.
- Generalized active target synchronization: Directory "src_RMA_GATS" has code with MPI one-sided implementation using generalized active target synshronization (Post-Start-Complete-Wait).
- Advanced Passive target synchronization: Directory "src_RMA_PTS" has code with MPI one-sided implementation using advanced passive target synchronization (Win_lock_all-Win_flush_all-Win_unlock_all.
 
Each of the above version has two implementation depending upon the whether MPI_Put or MPI_Get call is being used.

# Installation instructions: Pure MPI
- Go to the build directory "test/example1_no-omp" and clean previous build
	
        cd test/example1_no-omp
        make clean
- Baseline code will be compiled by default with no arguments to build script. Binary will be renamed to nekbone_baseline
	
        ./makenek-gcc
        
- Modified version can be compiled by providing two arguments to the build script. First argument can have three values: 0 for baseline, 1 for MPI_Put implementation and 2 for MPI_Get implementation. Second argument provides source directory with absolute path. Binaries will be named according to the arguments used with build script.
	
        ./makenek-gcc  <implementation: Use 1 for Put and 2 for Get> <source directory with absolute path>
	
	For e.g. below command will build a binary for MPI_Put implementation with code version src_RMA_GS and binary will be renamed as nekbone_RMA_GS_Put
	
		./makenek-gcc 1 src_RMA_GS
