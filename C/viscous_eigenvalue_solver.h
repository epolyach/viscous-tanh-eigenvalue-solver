#ifndef VISCOUS_EIGENVALUE_SOLVER_H
#define VISCOUS_EIGENVALUE_SOLVER_H

#include <mpfr.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gmp.h>

// Precision settings
#define PREC 1024  // Default precision in bits
#define DIGITS 68  // Number of digits in output CSV
#define DELTAC 4.2196  // Target control value for Delta/mu ratio

// Physical parameters (can be overridden via command line)
#define K_VALUE_DEFAULT "1.04"  // Default wavenumber as string
#define MU_DEFAULT "1e-9"      // Default viscosity as string
#define LANDAU_EIGENVALUE "0.0252105473484"  // Landau's eigenvalue as string
#define NMAX_DEFAULT 10000    // Default matrix size
#define MAX_ITER 100
#define ITER_TOL 1e-100

// Coefficient structures
typedef struct {
    mpfr_t *S_nm1;
    mpfr_t *S_np1;
} S1_Coeffs;

typedef struct {
    mpfr_t *S_nm2;
    mpfr_t *S_n0;
    mpfr_t *S_np2;
} S2_Coeffs;

typedef struct {
    mpfr_t *S_nm3;
    mpfr_t *S_nm1;
    mpfr_t *S_np1;
    mpfr_t *S_np3;
} S3_Coeffs;

typedef struct {
    mpfr_t *R_nm2;
    mpfr_t *R_n0;
    mpfr_t *R_np2;
} R_Coeffs;

typedef struct {
    mpfr_t *Q_nm3;
    mpfr_t *Q_nm1;
    mpfr_t *Q_np1;
    mpfr_t *Q_np3;
} Q_Coeffs;

typedef struct {
    mpfr_t *T_nm4;
    mpfr_t *T_nm2;
    mpfr_t *T_n0;
    mpfr_t *T_np2;
    mpfr_t *T_np4;
} T_Coeffs;

// Sparse matrix structure - using COO format for flexibility
typedef struct {
    long rows;
    long cols;
    long nnz;
    long capacity;
    long *row_idx;
    long *col_idx;
    mpfr_t *values;
} SparseMatrix;

// Structure to hold banded LU factorization (using LAPACK GB format)
typedef struct {
    long N;         // Dimension
    long kl;        // Number of lower diagonals
    long ku;        // Number of upper diagonals
    long lda;       // Leading dimension of ab (2*kl + ku + 1)
    mpfr_t *ab;     // Banded matrix storage (LAPACK format: (2*kl+ku+1) x N)
    long *ipiv;     // Pivot indices (0-based C index of the pivot row)
    mpfr_rnd_t rnd; // Rounding mode to use internally
} BandedLUFactors;



// Memory management for coefficient structures
void init_S1(S1_Coeffs *s1, long Nmax);
void init_S2(S2_Coeffs *s2, long Nmax);
void init_S3(S3_Coeffs *s3, long Nmax);
void init_R(R_Coeffs *r, long Nmax);
void init_Q(Q_Coeffs *q, long Nmax);
void init_T(T_Coeffs *t, long Nmax);

void free_S1(S1_Coeffs *s1, long Nmax);
void free_S2(S2_Coeffs *s2, long Nmax);
void free_S3(S3_Coeffs *s3, long Nmax);
void free_R(R_Coeffs *r, long Nmax);
void free_Q(Q_Coeffs *q, long Nmax);
void free_T(T_Coeffs *t, long Nmax);

// Coefficient calculation functions
void compute_S1(S1_Coeffs *s1, long Nmax, mpfr_rnd_t rnd);
void compute_S2(S2_Coeffs *s2, S1_Coeffs *s1, long Nmax, mpfr_rnd_t rnd);
void compute_S3(S3_Coeffs *s3, S2_Coeffs *s2, S1_Coeffs *s1, long Nmax, mpfr_rnd_t rnd);
void compute_R(R_Coeffs *r, S2_Coeffs *s2, long Nmax, mpfr_t k2, mpfr_rnd_t rnd);
void compute_Q(Q_Coeffs *q, S3_Coeffs *s3, S1_Coeffs *s1, long Nmax, mpfr_t k2, mpfr_rnd_t rnd);
void compute_T(T_Coeffs *t, S2_Coeffs *s2, R_Coeffs *r, long Nmax, mpfr_t k2, mpfr_rnd_t rnd);

// Sparse matrix operations
void init_sparse_matrix(SparseMatrix *matrix, long rows, long cols, long initial_capacity);
void sparse_matrix_add_element(SparseMatrix *matrix, long row, long col, mpfr_t value, mpfr_rnd_t rnd);
void free_sparse_matrix(SparseMatrix *matrix);
void sparse_matrix_vector_multiply(SparseMatrix *A, mpfr_t *x, mpfr_t *y, mpfr_rnd_t rnd);

// Matrix building functions
void build_matrices(SparseMatrix *R_mat, SparseMatrix *Q_mat, SparseMatrix *T_mat,
                   long Nmax, R_Coeffs *r_coeffs, Q_Coeffs *q_coeffs, T_Coeffs *t_coeffs,
                   mpfr_rnd_t rnd);

// Linear Algebra Utilities
void normalize_vector(mpfr_t *v, long n, mpfr_rnd_t rnd);
void rayleigh_quotient(SparseMatrix *A, SparseMatrix *B, mpfr_t *x, mpfr_t result, long n, mpfr_rnd_t rnd);

// Banded Matrix Operations
void init_banded_lu_factors(BandedLUFactors *lu_factors, long N, long kl, long ku, mpfr_rnd_t rnd);
void free_banded_lu_factors(BandedLUFactors *lu_factors);
int banded_lu_factorization_no_pivot(SparseMatrix *K, BandedLUFactors *lu_factors);
int solve_banded_lu_no_pivot(BandedLUFactors *lu_factors, mpfr_t *b, mpfr_t *x);
void build_K_matrix(SparseMatrix *K, SparseMatrix *M, SparseMatrix *R, mpfr_t sigma, long Nmax, mpfr_rnd_t rnd);

// Eigenvalue solver
void find_eigenvalue(SparseMatrix *M, SparseMatrix *R, mpfr_t *eigenvalue, mpfr_t *eigenvector,
                     long Nmax, mpfr_t target_sigma, mpfr_t mu, mpfr_rnd_t rnd);

// Command line parsing
void parse_args(int argc, char **argv, mpfr_t k_value, mpfr_t mu, mpfr_t target, long *Nmax, mpfr_rnd_t rnd);

void save_eigen_data(mpfr_t eigenvalue, mpfr_t *eigenvector, long Nmax, mpfr_t k_value, 
                     mpfr_t mu, int precision);

void create_M_matrix_optimized(SparseMatrix *M_mat, SparseMatrix *Q_mat, SparseMatrix *T_mat, 
                             mpfr_t mu, mpfr_rnd_t rnd);

void build_K_matrix_optimized(SparseMatrix *K, SparseMatrix *M, SparseMatrix *R, mpfr_t sigma, long Nmax, mpfr_rnd_t rnd);

void build_K_matrix_with_progress(SparseMatrix *K, SparseMatrix *M, SparseMatrix *R, mpfr_t sigma, long Nmax, mpfr_rnd_t rnd);

#endif // VISCOUS_EIGENVALUE_SOLVER_H
