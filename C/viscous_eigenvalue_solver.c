#include "viscous_eigenvalue_solver.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>

#define MPFR_ZERO_TOL 1e-100 // Small tolerance for checking singularity
#define BANDED_INDEX(i, j, kl, ku, lda) ((j) * (lda) + (kl + ku + i - j))

// Parse command line arguments
void parse_args(int argc, char **argv, mpfr_t k_value, mpfr_t mu, mpfr_t target, long *Nmax, mpfr_rnd_t rnd) {
    // Set defaults using string constants
    mpfr_set_str(k_value, K_VALUE_DEFAULT, 10, rnd);
    mpfr_set_str(mu, MU_DEFAULT, 10, rnd);
    mpfr_set_str(target, LANDAU_EIGENVALUE, 10, rnd);
    
    *Nmax = NMAX_DEFAULT;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-k") == 0 && i+1 < argc) {
            mpfr_set_str(k_value, argv[++i], 10, rnd);
        } else if (strcmp(argv[i], "-mu") == 0 && i+1 < argc) {
            mpfr_set_str(mu, argv[++i], 10, rnd);
        } else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) {
            mpfr_set_str(target, argv[++i], 10, rnd);
        } else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            *Nmax = atol(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -k <value>    Wavenumber (default: %s)\n", K_VALUE_DEFAULT);
            printf("  -mu <value>   Viscosity (default: %s)\n", MU_DEFAULT);
            printf("  -t <value>    Landau eigenvalue (default: %s)\n", LANDAU_EIGENVALUE);
            printf("  -n <value>    Matrix size (default: %d)\n", NMAX_DEFAULT);
            printf("  -h, --help    Display this help message\n");
            exit(0);
        }
    }
    
    // Validate parameters
    if (*Nmax <= 0) {
        printf("Error: Matrix size (Nmax) must be positive\n");
        exit(1);
    }
}

// Initialization of banded LU factors
void init_banded_lu_factors(BandedLUFactors *lu_factors, long N, long kl, long ku, mpfr_rnd_t rnd) {
    lu_factors->N = N; lu_factors->kl = kl; lu_factors->ku = ku;
    lu_factors->lda = 2 * kl + ku + 1;
    lu_factors->ab = NULL; lu_factors->ipiv = NULL; lu_factors->rnd = rnd;
    if (N <= 0) return;
    size_t num_elements_ab = (size_t)N * lu_factors->lda;
    lu_factors->ab = (mpfr_t *)malloc(num_elements_ab * sizeof(mpfr_t));
    
    if (!lu_factors->ab) {
        fprintf(stderr, "Error: Failed to allocate memory for banded LU factors (ab) size %zu.\n", num_elements_ab);
        lu_factors->N = 0;
        exit(EXIT_FAILURE);
    }
    
    for (size_t i = 0; i < num_elements_ab; i++) { 
        mpfr_init2(lu_factors->ab[i], PREC); 
        mpfr_set_zero(lu_factors->ab[i], 0); 
    }

    lu_factors->ipiv = (long *)malloc((size_t)N * sizeof(long));
    
    if (!lu_factors->ipiv) {
        fprintf(stderr, "Error: Failed to allocate memory for pivot indices (ipiv) size %ld.\n", N);
        for (size_t i = 0; i < num_elements_ab; i++) mpfr_clear(lu_factors->ab[i]);
        free(lu_factors->ab);
        lu_factors->ab = NULL;
        lu_factors->N = 0;
        exit(EXIT_FAILURE);
    }
    
    for(long i=0; i<N; ++i) lu_factors->ipiv[i] = i;
}

// Free banded LU factors
void free_banded_lu_factors(BandedLUFactors *lu_factors) {
    if (!lu_factors) return;
    if (lu_factors->ab) {
        size_t num_elements_ab = (size_t)lu_factors->N * lu_factors->lda;
         if (lu_factors->N > 0 && lu_factors->lda > 0) {
            for (size_t i = 0; i < num_elements_ab; i++) {
                mpfr_clear(lu_factors->ab[i]);
            }
         }
        free(lu_factors->ab);
        lu_factors->ab = NULL;
    }
    if (lu_factors->ipiv) {
        free(lu_factors->ipiv);
        lu_factors->ipiv = NULL;
    }
    lu_factors->N = 0;
    lu_factors->kl = 0;
    lu_factors->ku = 0;
    lu_factors->lda = 0;
}

// Sparse matrix-vector product: y = A*x
void sparse_matrix_vector_multiply(SparseMatrix *A, mpfr_t *x, mpfr_t *y, mpfr_rnd_t rnd) {
    if (!A || !x || !y || !A->values || !A->row_idx || !A->col_idx) {
         fprintf(stderr, "Error: Invalid arguments passed to sparse_matrix_vector_multiply.\n");
         for (long i = 0; i < A->rows; i++) { mpfr_set_nan(y[i]); }
         return;
    }

    for (long i = 0; i < A->rows; i++) {
        mpfr_set_zero(y[i], 1);
    }

    mpfr_t temp;
    mpfr_init2(temp, PREC);

    for (long k = 0; k < A->nnz; k++) {
        long i = A->row_idx[k];
        long j = A->col_idx[k];

        if (i < 0 || i >= A->rows) {
            fprintf(stderr, "ERROR: Sparse matrix multiply: row index i=%ld out of bounds [0, %ld) at k=%ld.\n", i, A->rows, k);
            mpfr_clear(temp);
            for (long row_idx = 0; row_idx < A->rows; row_idx++) { mpfr_set_nan(y[row_idx]); }
            return;
        }
        if (j < 0 || j >= A->cols) {
             fprintf(stderr, "ERROR: Sparse matrix multiply: col index j=%ld out of bounds [0, %ld) at k=%ld.\n", j, A->cols, k);
             mpfr_clear(temp);
             for (long row_idx = 0; row_idx < A->rows; row_idx++) { mpfr_set_nan(y[row_idx]); }
             return;
        }

        mpfr_mul(temp, A->values[k], x[j], rnd);
        mpfr_add(y[i], y[i], temp, rnd);
    }

    mpfr_clear(temp);
}

// Normalize a vector to have unit 2-norm
void normalize_vector(mpfr_t *v, long n, mpfr_rnd_t rnd) {
     if (!v || n <= 0) return;

    mpfr_t norm_sq, norm, temp;
    mpfr_init2(norm_sq, PREC);
    mpfr_init2(norm, PREC);
    mpfr_init2(temp, PREC);

    mpfr_set_zero(norm_sq, 1);
    for (long i = 0; i < n; i++) {
        mpfr_sqr(temp, v[i], rnd);
        mpfr_add(norm_sq, norm_sq, temp, rnd);
    }

    mpfr_sqrt(norm, norm_sq, rnd);

    if (!mpfr_zero_p(norm)) {
        for (long i = 0; i < n; i++) {
            mpfr_div(v[i], v[i], norm, rnd);
        }
    } else {
        fprintf(stderr, "Warning: Vector has zero norm in normalize_vector, cannot normalize.\n");
    }

    mpfr_clear(norm_sq);
    mpfr_clear(norm);
    mpfr_clear(temp);
}

// Compute Rayleigh quotient: lambda = (x'*A*x)/(x'*B*x)
void rayleigh_quotient(SparseMatrix *A, SparseMatrix *B, mpfr_t *x, mpfr_t result, long n, mpfr_rnd_t rnd) {
     if (!A || !B || !x || !result || n <= 0) {
         fprintf(stderr, "Error: Invalid arguments to rayleigh_quotient.\n");
         mpfr_set_nan(result);
         return;
     }

    mpfr_t *Ax = NULL;
    mpfr_t *Bx = NULL;
    Ax = (mpfr_t *)malloc(n * sizeof(mpfr_t));
    Bx = (mpfr_t *)malloc(n * sizeof(mpfr_t));
    if (!Ax || !Bx) {
         fprintf(stderr, "Error: Memory allocation failed in rayleigh_quotient.\n");
         free(Ax); free(Bx);
         mpfr_set_nan(result);
         return;
    }

    for (long i = 0; i < n; i++) {
        mpfr_init2(Ax[i], PREC);
        mpfr_init2(Bx[i], PREC);
    }

    sparse_matrix_vector_multiply(A, x, Ax, rnd);
    sparse_matrix_vector_multiply(B, x, Bx, rnd);

    mpfr_t numerator, denominator, temp_dot;
    mpfr_init2(numerator, PREC);
    mpfr_init2(denominator, PREC);
    mpfr_init2(temp_dot, PREC);

    mpfr_set_zero(numerator, 1);
    mpfr_set_zero(denominator, 1);

    for (long i = 0; i < n; i++) {
        mpfr_mul(temp_dot, x[i], Ax[i], rnd);
        mpfr_add(numerator, numerator, temp_dot, rnd);

        mpfr_mul(temp_dot, x[i], Bx[i], rnd);
        mpfr_add(denominator, denominator, temp_dot, rnd);
    }

    if (!mpfr_zero_p(denominator)) {
        mpfr_div(result, numerator, denominator, rnd);
    } else {
         fprintf(stderr, "Warning: Zero denominator in initial Rayleigh quotient.\n");
         mpfr_set_nan(result);
    }

    for (long i = 0; i < n; i++) {
        mpfr_clear(Ax[i]);
        mpfr_clear(Bx[i]);
    }
    free(Ax);
    free(Bx);

    mpfr_clear(numerator);
    mpfr_clear(denominator);
    mpfr_clear(temp_dot);
}

// Build K = M - sigma*R matrix
void build_K_matrix(SparseMatrix *K, SparseMatrix *M, SparseMatrix *R, mpfr_t sigma, long Nmax, mpfr_rnd_t rnd) {
    printf("Building K = M - sigma*R matrix...\n");
    long estimated_nnz = M->nnz;
    init_sparse_matrix(K, Nmax, Nmax, estimated_nnz > 0 ? estimated_nnz : 10);

    mpfr_t neg_sigma_R_val;
    mpfr_init2(neg_sigma_R_val, PREC);

    for (long i = 0; i < M->nnz; ++i) {
        sparse_matrix_add_element(K, M->row_idx[i], M->col_idx[i], M->values[i], rnd);
    }

    for (long i = 0; i < R->nnz; ++i) {
        mpfr_mul(neg_sigma_R_val, R->values[i], sigma, rnd);
        mpfr_neg(neg_sigma_R_val, neg_sigma_R_val, rnd);
        sparse_matrix_add_element(K, R->row_idx[i], R->col_idx[i], neg_sigma_R_val, rnd);
    }

    mpfr_clear(neg_sigma_R_val);
    printf("Finished building K matrix. NNZ count in K: %ld\n", K->nnz);
}

// LU factorization without pivoting
int banded_lu_factorization_no_pivot(SparseMatrix *K, BandedLUFactors *lu_factors) {
    long N = lu_factors->N;
    long kl = lu_factors->kl;
    long ku = lu_factors->ku;
    long lda = lu_factors->lda;
    mpfr_t *ab = lu_factors->ab;
    mpfr_rnd_t rnd = lu_factors->rnd;

    if (K->rows != N || K->cols != N) {
        fprintf(stderr, "Error: Matrix dimensions don't match in banded_lu_factorization_no_pivot\n");
        return -1;
    }
    if (!ab) {
        fprintf(stderr, "Error: Null pointers (ab) in banded_lu_factorization_no_pivot\n");
        return -1;
    }
    if (N <= 0) return 0;

    size_t num_elements_ab = (size_t)N * lda;

    printf("Copying sparse K to banded format 'ab' (for no-pivot LU)...\n");
    for (size_t i = 0; i < num_elements_ab; i++) {
        mpfr_set_zero(ab[i], 0);
    }
    for (long k_idx = 0; k_idx < K->nnz; ++k_idx) {
        long i = K->row_idx[k_idx];
        long j = K->col_idx[k_idx];
        if (i >= 0 && i < N && j >= 0 && j < N && (i - j) <= kl && (j - i) <= ku) {
            long row_in_band = kl + ku + i - j;
            if (row_in_band >= 0 && row_in_band < lda) {
                size_t index_in_ab = BANDED_INDEX(row_in_band, j, kl, ku, lda);
                if (index_in_ab < num_elements_ab) {
                    mpfr_add(ab[index_in_ab], ab[index_in_ab], K->values[k_idx], rnd);
                }
            }
        }
    }
    printf("Finished copying K to 'ab'.\n");

    mpfr_t temp, pivot_val, abs_pivot_val, multiplier, eps_threshold;
    mpfr_init2(temp, PREC);
    mpfr_init2(pivot_val, PREC);
    mpfr_init2(abs_pivot_val, PREC);
    mpfr_init2(multiplier, PREC);
    mpfr_init2(eps_threshold, PREC);
    mpfr_set_d(eps_threshold, 1e-50, rnd);

    printf("Starting banded LU factorization WITHOUT pivoting...\n");
    for (long j = 0; j < N; j++) {
        size_t pivot_idx = BANDED_INDEX(kl + ku, j, kl, ku, lda);

        if (pivot_idx >= num_elements_ab) {
            fprintf(stderr, "Error: Diagonal pivot index %zu out of bounds for column j=%ld.\n", pivot_idx, j);
            mpfr_clear(temp); mpfr_clear(pivot_val); mpfr_clear(abs_pivot_val);
            mpfr_clear(multiplier); mpfr_clear(eps_threshold);
            return -3;
        }
        mpfr_set(pivot_val, ab[pivot_idx], rnd);

        mpfr_abs(abs_pivot_val, pivot_val, rnd);
        if (mpfr_cmp(abs_pivot_val, eps_threshold) < 0) {
            fprintf(stderr, "ERROR (No Pivot LU): Near-zero diagonal element K(%ld, %ld) = ", j, j);
            mpfr_printf("%.6Re encountered.\n", pivot_val);
            fprintf(stderr, "  LU factorization without pivoting cannot proceed.\n");
            mpfr_clear(temp); mpfr_clear(pivot_val); mpfr_clear(abs_pivot_val);
            mpfr_clear(multiplier); mpfr_clear(eps_threshold);
            return j + 1;
        }

        for (long i = j + 1; i < N && i <= j + kl; i++) {
            long row_i_in_band = kl + ku + i - j;
            if (row_i_in_band < 0 || row_i_in_band >= lda) continue;

            size_t element_below_pivot_idx = BANDED_INDEX(row_i_in_band, j, kl, ku, lda);
            if (element_below_pivot_idx >= num_elements_ab) continue;

            if (!mpfr_zero_p(ab[element_below_pivot_idx])) {
                mpfr_div(multiplier, ab[element_below_pivot_idx], pivot_val, rnd);
                mpfr_set(ab[element_below_pivot_idx], multiplier, rnd);

                for (long k_col = j + 1; k_col < N && k_col <= j + ku; ++k_col) {
                    long row_ik_in_band = kl + ku + i - k_col;
                    if (row_ik_in_band < 0 || row_ik_in_band >= lda) continue;
                    size_t ik_idx = BANDED_INDEX(row_ik_in_band, k_col, kl, ku, lda);
                    if (ik_idx >= num_elements_ab) continue;

                    long row_jk_in_band = kl + ku + j - k_col;
                    if (row_jk_in_band < 0 || row_jk_in_band >= lda) continue;
                    size_t jk_idx = BANDED_INDEX(row_jk_in_band, k_col, kl, ku, lda);
                    if (jk_idx >= num_elements_ab) continue;

                    mpfr_mul(temp, multiplier, ab[jk_idx], rnd);
                    mpfr_sub(ab[ik_idx], ab[ik_idx], temp, rnd);
                }
            } else {
                 mpfr_set_zero(ab[element_below_pivot_idx], 0);
            }
        }
    }

    printf("Finished banded LU factorization WITHOUT pivoting.\n");

    mpfr_clear(temp); mpfr_clear(pivot_val); mpfr_clear(abs_pivot_val);
    mpfr_clear(multiplier); mpfr_clear(eps_threshold);

    return 0;
}

// Solve banded LU without pivoting
int solve_banded_lu_no_pivot(BandedLUFactors *lu_factors, mpfr_t *b, mpfr_t *x) {
    long N = lu_factors->N;
    long kl = lu_factors->kl;
    long ku = lu_factors->ku;
    long lda = lu_factors->lda;
    mpfr_t *ab = lu_factors->ab;
    mpfr_rnd_t rnd = lu_factors->rnd;

    if (N <= 0) return 0;
    if (!ab) {
        fprintf(stderr, "Error: Invalid LU factors (ab is NULL) in solve_banded_lu_no_pivot\n");
        return -1;
    }

    size_t num_elements_ab = (size_t)N * lda;

    mpfr_t temp_sum;
    mpfr_init2(temp_sum, PREC);

    if (x != b) {
        for (long i = 0; i < N; ++i) mpfr_set(x[i], b[i], rnd);
    }

    for (long i = 0; i < N; ++i) {
        mpfr_set(temp_sum, x[i], rnd);
        for (long j = i - kl > 0 ? i - kl : 0; j < i; ++j) {
            long row_Lij_in_band = kl + ku + i - j;
            if (row_Lij_in_band >= 0 && row_Lij_in_band < lda) {
                size_t Lij_idx = BANDED_INDEX(row_Lij_in_band, j, kl, ku, lda);
                if (Lij_idx < num_elements_ab) {
                    mpfr_t term;
                    mpfr_init2(term, PREC);
                    mpfr_mul(term, ab[Lij_idx], x[j], rnd);
                    mpfr_sub(temp_sum, temp_sum, term, rnd);
                    mpfr_clear(term);
                }
            }
        }
        mpfr_set(x[i], temp_sum, rnd);
    }

    for (long i = N - 1; i >= 0; --i) {
        mpfr_set(temp_sum, x[i], rnd);

        for (long j = i + 1; j < N && j <= i + ku; ++j) {
            long row_Uij_in_band = kl + ku + i - j;
            if (row_Uij_in_band >= 0 && row_Uij_in_band < lda) {
                size_t Uij_idx = BANDED_INDEX(row_Uij_in_band, j, kl, ku, lda);
                if (Uij_idx < num_elements_ab) {
                    mpfr_t term;
                    mpfr_init2(term, PREC);
                    mpfr_mul(term, ab[Uij_idx], x[j], rnd);
                    mpfr_sub(temp_sum, temp_sum, term, rnd);
                    mpfr_clear(term);
                }
            }
        }

        size_t Uii_idx = BANDED_INDEX(kl + ku, i, kl, ku, lda);
        if (Uii_idx >= num_elements_ab) {
            fprintf(stderr, "Error: Invalid diagonal index %zu in backward solve (no pivot) at i=%ld\n", Uii_idx, i);
            mpfr_clear(temp_sum);
            return -3;
        }

        mpfr_t pivot_abs, eps_tol;
        mpfr_init2(pivot_abs, PREC);
        mpfr_init2(eps_tol, PREC);
        mpfr_set_d(eps_tol, 1e-50, rnd);
        mpfr_abs(pivot_abs, ab[Uii_idx], rnd);

        if (mpfr_cmp(pivot_abs, eps_tol) < 0) {
            fprintf(stderr, "ERROR (No Pivot Solve): Near-zero diagonal U(%ld,%ld) = ", i, i);
            mpfr_printf("%.6Re encountered.\n", ab[Uii_idx]);
            mpfr_clear(pivot_abs); mpfr_clear(eps_tol); mpfr_clear(temp_sum);
            return -4;
        }

        mpfr_div(x[i], temp_sum, ab[Uii_idx], rnd);
        mpfr_clear(pivot_abs);
        mpfr_clear(eps_tol);

        if (!mpfr_number_p(x[i])) {
             fprintf(stderr, "ERROR: Non-finite result (Inf/NaN) in backward solve (no pivot) at row %ld\n", i);
             mpfr_clear(temp_sum);
             return -5;
        }
    }

    mpfr_clear(temp_sum);
    return 0;
}

// Find eigenvalue using inverse iteration
void find_eigenvalue(SparseMatrix *M, SparseMatrix *R, mpfr_t *eigenvalue, mpfr_t *eigenvector,
    long Nmax, mpfr_t target_sigma, mpfr_t mu, mpfr_rnd_t rnd) {
    
    mpfr_t temp, deltac;
    mpfr_init2(temp, PREC); 
    mpfr_init2(deltac, PREC);
    const int max_iter = MAX_ITER;
    const double tol = ITER_TOL;

    printf("\n--- Starting Shift-and-Invert Inverse Iteration ---\n");
    
    // Calculate shift (target_sigma)
    mpfr_printf("Landau eigenvalue (base for shift) = %.15RNf\n", target_sigma);
    mpfr_set_d(deltac, DELTAC, rnd);
    mpfr_mul(temp, mu, deltac, rnd);
    mpfr_add(target_sigma, target_sigma, temp, rnd);
    mpfr_printf("Theoretical target = %.15RNf\n", target_sigma);
    
    mpfr_clear(temp); 
    mpfr_clear(deltac);

    // Allocate vectors needed for the inverse iteration
    mpfr_t *v = NULL;
    mpfr_t *Rv = NULL;
    mpfr_t *Mv = NULL;
    mpfr_t *w = NULL;
    
    v = (mpfr_t *)malloc(Nmax * sizeof(mpfr_t));
    Rv = (mpfr_t *)malloc(Nmax * sizeof(mpfr_t));
    Mv = (mpfr_t *)malloc(Nmax * sizeof(mpfr_t));
    w = (mpfr_t *)malloc(Nmax * sizeof(mpfr_t));
    
    if (!v || !Rv || !Mv || !w) {
        fprintf(stderr, "Error: Memory allocation failed in find_eigenvalue\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize MPFR variables
    mpfr_t lambda_old, lambda, lambda_diff, shifted_lambda;
    mpfr_t numer, denom, temp_norm_check;
    
    mpfr_init2(lambda_old, PREC);
    mpfr_init2(lambda, PREC);
    mpfr_init2(lambda_diff, PREC);
    mpfr_init2(shifted_lambda, PREC);
    mpfr_init2(numer, PREC);
    mpfr_init2(denom, PREC);
    mpfr_init2(temp_norm_check, PREC);
    
    // Set up the K matrix and LU factorization
    SparseMatrix K_mat;
    BandedLUFactors lu_factors;
    long kl = 4;  // Lower bandwidth
    long ku = 4;  // Upper bandwidth
    
    init_banded_lu_factors(&lu_factors, Nmax, kl, ku, rnd);
    if (lu_factors.N == 0) {
        fprintf(stderr, "Error: Failed to initialize LU factors\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize all vectors
    for (long i = 0; i < Nmax; i++) {
        mpfr_init2(v[i], PREC);
        mpfr_init2(Rv[i], PREC);
        mpfr_init2(Mv[i], PREC);
        mpfr_init2(w[i], PREC);
    }
    
    // Generate initial random vector
    printf("Initializing guess vector 'v' with random values...\n");
    gmp_randstate_t rand_state;
    gmp_randinit_default(rand_state);
    gmp_randseed_ui(rand_state, (unsigned long)time(NULL));
    
    for (long i = 0; i < Nmax; i++) {
        mpfr_urandomb(v[i], rand_state);
    }
    gmp_randclear(rand_state);
    
    // Normalize the initial vector
    normalize_vector(v, Nmax, rnd);
    printf("Initial random vector normalized.\n");
    
    // Build K = M - sigma*R
    // build_K_matrix(&K_mat, M, R, target_sigma, Nmax, rnd);
    // build_K_matrix_optimized(&K_mat, M, R, target_sigma, Nmax, rnd);
    build_K_matrix_with_progress(&K_mat, M, R, target_sigma, Nmax, rnd);
    
    // Perform LU factorization
    printf("Performing LU factorization of K...\n");
    clock_t lu_start = clock();
    int lu_status = banded_lu_factorization_no_pivot(&K_mat, &lu_factors);
    clock_t lu_end = clock();
    printf("LU factorization completed (status: %d) in %.2f seconds.\n",
           lu_status, (double)(lu_end - lu_start) / CLOCKS_PER_SEC);

    // Error handling for lu_factorization
    if (lu_status != 0) {
        fprintf(stderr, "Error: Banded LU factorization failed (status: %d). Matrix appears to be singular.\n", lu_status);
        fprintf(stderr, "Try adjusting the shift value or checking the matrix construction.\n");
        
        // Set eigenvalue to NaN to indicate failure
        mpfr_set_nan(*eigenvalue);
        
        // Clean up and exit
        free_banded_lu_factors(&lu_factors);
        free_sparse_matrix(&K_mat);
        for (long i = 0; i < Nmax; i++) {
            mpfr_clear(v[i]); mpfr_clear(Rv[i]); mpfr_clear(Mv[i]); mpfr_clear(w[i]);
        }
        free(v); free(Rv); free(Mv); free(w);
        mpfr_clear(lambda_old); mpfr_clear(lambda); mpfr_clear(lambda_diff);
        mpfr_clear(shifted_lambda); mpfr_clear(numer); mpfr_clear(denom);
        mpfr_clear(temp_norm_check);
        return;
    }
    
    // Calculate initial Rayleigh quotient using proper approach for shift-invert
    sparse_matrix_vector_multiply(M, v, Mv, rnd);
    sparse_matrix_vector_multiply(R, v, Rv, rnd);
    
    // Initialize accumulators
    mpfr_set_zero(numer, 1);
    mpfr_set_zero(denom, 1);
    mpfr_t temp_prod;
    mpfr_init2(temp_prod, PREC);
    
    // Calculate numer = v'*M*v, denom = v'*R*v
    for (long i = 0; i < Nmax; i++) {
        // numer += v[i] * Mv[i]
        mpfr_mul(temp_prod, v[i], Mv[i], rnd);
        mpfr_add(numer, numer, temp_prod, rnd);
        
        // denom += v[i] * Rv[i]
        mpfr_mul(temp_prod, v[i], Rv[i], rnd);
        mpfr_add(denom, denom, temp_prod, rnd);
    }
    
    // Calculate lambda_old = numer / denom (standard Rayleigh quotient)
    if (!mpfr_zero_p(denom)) {
        mpfr_div(lambda_old, numer, denom, rnd);
    } else {
        fprintf(stderr, "Warning: Zero denominator in initial Rayleigh quotient.\n");
        mpfr_set_d(lambda_old, 0.0, rnd); // Default value
    }
    
    mpfr_printf("Initial Rayleigh quotient lambda_old = %.15RNf\n", lambda_old);
    
    // Print header for iteration output
    printf("\nIter    Eigenvalue           Rel. Change      ||w||_inf\n");
    printf("----------------------------------------------------------\n");
    
    // Inverse Iteration Loop
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        // 1. Calculate R*v
        sparse_matrix_vector_multiply(R, v, Rv, rnd);
        
        // 2. Solve (M - sigma*R)*w = R*v
        int solve_status = solve_banded_lu_no_pivot(&lu_factors, Rv, w);
        if (solve_status != 0) {
            fprintf(stderr, "Error: LU solve failed in iteration %d (status: %d)\n", iter+1, solve_status);
            
            // Set eigenvalue to NaN to indicate failure
            mpfr_set_nan(*eigenvalue);
            
            // Clean up and exit
            mpfr_clear(temp_prod);
            free_banded_lu_factors(&lu_factors);
            free_sparse_matrix(&K_mat);
            for (long i = 0; i < Nmax; i++) {
                mpfr_clear(v[i]); mpfr_clear(Rv[i]); mpfr_clear(Mv[i]); mpfr_clear(w[i]);
            }
            free(v); free(Rv); free(Mv); free(w);
            mpfr_clear(lambda_old); mpfr_clear(lambda); mpfr_clear(lambda_diff);
            mpfr_clear(shifted_lambda); mpfr_clear(numer); mpfr_clear(denom);
            mpfr_clear(temp_norm_check);
            return;
        }
        
        // 3. Calculate |w|_inf for reporting
        mpfr_set_zero(temp_norm_check, 0);
        for (long i = 0; i < Nmax; i++) {
            mpfr_abs(lambda_diff, w[i], rnd);
            if (mpfr_cmp(lambda_diff, temp_norm_check) > 0) {
                mpfr_set(temp_norm_check, lambda_diff, rnd);
            }
        }
        
        // 4. Normalize w
        normalize_vector(w, Nmax, rnd);
        
        // 5. Set v = w for next iteration
        for (long i = 0; i < Nmax; i++) {
            mpfr_set(v[i], w[i], rnd);
        }
        
        // 6. Calculate new Rayleigh quotient
        sparse_matrix_vector_multiply(M, v, Mv, rnd);
        sparse_matrix_vector_multiply(R, v, Rv, rnd);
        
        // Reset accumulators
        mpfr_set_zero(numer, 1);
        mpfr_set_zero(denom, 1);
        
        // Calculate numer = v'*M*v, denom = v'*R*v
        for (long i = 0; i < Nmax; i++) {
            mpfr_mul(temp_prod, v[i], Mv[i], rnd);
            mpfr_add(numer, numer, temp_prod, rnd);
            
            mpfr_mul(temp_prod, v[i], Rv[i], rnd);
            mpfr_add(denom, denom, temp_prod, rnd);
        }
        
        // Calculate lambda = numer / denom
        if (!mpfr_zero_p(denom)) {
            mpfr_div(lambda, numer, denom, rnd);
        } else {
            fprintf(stderr, "Warning: Zero denominator in Rayleigh quotient (iter %d).\n", iter+1);
            mpfr_set(lambda, lambda_old, rnd); // Keep previous value
            break;
        }
        
        // 7. Calculate relative change and check convergence
        mpfr_sub(lambda_diff, lambda, lambda_old, rnd);
        mpfr_abs(lambda_diff, lambda_diff, rnd);
        mpfr_abs(temp_norm_check, lambda, rnd);
        
        double rel_change_d = 0.0;
        if (mpfr_cmp_d(temp_norm_check, 1e-30) > 0) {
            mpfr_div(lambda_diff, lambda_diff, temp_norm_check, rnd);
            rel_change_d = mpfr_get_d(lambda_diff, MPFR_RNDN);
        } else if (mpfr_zero_p(lambda_diff)) {
            rel_change_d = 0.0;
        } else {
            rel_change_d = 1.0;
        }
        
        // Print progress
        mpfr_printf("%4d    %.15RNf    %.8e    %.8Re\n", iter + 1, lambda, rel_change_d, temp_norm_check);
        
        // Check for convergence
        if (fabs(rel_change_d) < tol) {
            printf("\nConverged after %d iterations.\n", iter + 1);
            break;
        }
        
        // Update lambda_old for next iteration
        mpfr_set(lambda_old, lambda, rnd);
    }
    
    // Check if max iterations reached
    if (iter == max_iter) {
        printf("\nWarning: Maximum iterations (%d) reached without convergence.\n", max_iter);
    }
    
    // Set the eigenvalue
    mpfr_set(*eigenvalue, lambda, rnd);
    
    // Copy the eigenvector to output
    for (long i = 0; i < Nmax; i++) {
        mpfr_set(eigenvector[i], v[i], rnd);
    }
    
    // Clean up
    printf("Cleaning up eigenvalue solver resources...\n");
    mpfr_clear(temp_prod);
    mpfr_clear(numer);
    mpfr_clear(denom);
    mpfr_clear(shifted_lambda);
    mpfr_clear(lambda_old);
    mpfr_clear(lambda);
    mpfr_clear(lambda_diff);
    mpfr_clear(temp_norm_check);
    
    free_banded_lu_factors(&lu_factors);
    free_sparse_matrix(&K_mat);
    
    for (long i = 0; i < Nmax; i++) {
        mpfr_clear(v[i]);
        mpfr_clear(Rv[i]);
        mpfr_clear(Mv[i]);
        mpfr_clear(w[i]);
    }
    free(v);
    free(Rv);
    free(Mv);
    free(w);
    
    printf("--- Finished Shift-and-Invert Inverse Iteration ---\n");
}

void save_eigen_data(mpfr_t eigenvalue, mpfr_t *eigenvector, long Nmax, mpfr_t k_value, 
                     mpfr_t mu, int precision) {
    // Create filename
    char filename[256];
    char k_str[64], mu_str[64];
    
    // Convert k_value and mu to strings with scientific notation
    mpfr_sprintf(k_str, "%.6Rg", k_value);
    mpfr_sprintf(mu_str, "%.6Rg", mu);
    
    int digits = (int)(DIGITS); 

    // Format the filename
    snprintf(filename, sizeof(filename), "EVC_k=%s_mu=%s_Nmax=%ld_Precision_=%d_Digits=%d.dat", 
             k_str, mu_str, Nmax, precision, digits);
    
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    printf("Saving eigenvalue and eigenvector to %s...\n", filename);
    
    // Calculate how many digits to print based on precision
    // int digits = (int)(precision * 0.301); // log10(2) ≈ 0.301, so precision*0.301 gives decimal digits
    
    // Allocate a buffer for the string representation
    char *buffer = (char*)malloc((digits + 50) * sizeof(char)); // Extra space for scientific notation
    if (!buffer) {
        fprintf(stderr, "Error: Memory allocation failed for string buffer\n");
        fclose(fp);
        return;
    }
    
    // Write eigenvalue with full precision
    mpfr_sprintf(buffer, "%.*Rg", digits, eigenvalue);
    fprintf(fp, "%s\n", buffer);
    
    // Create a normalization factor (1/eigenvector[0])
    mpfr_t norm_factor;
    mpfr_init2(norm_factor, precision);
    
    // Check if eigenvector[0] is zero to avoid division by zero
    if (mpfr_zero_p(eigenvector[0])) {
        fprintf(stderr, "Warning: Cannot normalize by eigenvector[0] as it is zero.\n");
        fprintf(stderr, "Writing original eigenvector without normalization.\n");
        
        // Write eigenvector elements with full precision without normalization
        for (long i = 0; i < Nmax; i++) {
            mpfr_sprintf(buffer, "%.*Rg", digits, eigenvector[i]);
            fprintf(fp, "%s\n", buffer);
        }
    } else {
        // Calculate normalization factor as 1/eigenvector[0]
        mpfr_ui_div(norm_factor, 1, eigenvector[0], MPFR_RNDN);
        
        // Write normalized eigenvector elements with full precision
        mpfr_t normalized_value;
        mpfr_init2(normalized_value, precision);
        
        for (long i = 0; i < Nmax; i++) {
            // Multiply each element by the normalization factor
            mpfr_mul(normalized_value, eigenvector[i], norm_factor, MPFR_RNDN);
            mpfr_sprintf(buffer, "%.*Rg", digits, normalized_value);
            fprintf(fp, "%s\n", buffer);
        }
        
        mpfr_clear(normalized_value);
    }
    
    // Clean up
    mpfr_clear(norm_factor);
    free(buffer);
    fclose(fp);
    printf("Eigenvalue and normalized eigenvector saved successfully to %s\n", filename);
}

void create_M_matrix_optimized(SparseMatrix *M_mat, SparseMatrix *Q_mat, SparseMatrix *T_mat, 
                             mpfr_t mu, mpfr_rnd_t rnd) {
    
    // Calculate exact number of non-zeros for M
    long M_nnz = Q_mat->nnz + T_mat->nnz;
    
    // Initialize M with exact capacity - no need to resize later
    init_sparse_matrix(M_mat, Q_mat->rows, Q_mat->cols, M_nnz);
    
    mpfr_t scaled_value;
    mpfr_init2(scaled_value, PREC);
    
    // Copy Q elements directly (no search needed)
    for (long i = 0; i < Q_mat->nnz; i++) {
        // Just copy the element - no need to search
        M_mat->row_idx[M_mat->nnz] = Q_mat->row_idx[i];
        M_mat->col_idx[M_mat->nnz] = Q_mat->col_idx[i];
        mpfr_set(M_mat->values[M_mat->nnz], Q_mat->values[i], rnd);
        M_mat->nnz++;
    }
    
    // Add mu*T elements directly (no search needed)
    for (long i = 0; i < T_mat->nnz; i++) {
        mpfr_mul(scaled_value, T_mat->values[i], mu, rnd);
        
        // Just add the element - no search needed
        M_mat->row_idx[M_mat->nnz] = T_mat->row_idx[i];
        M_mat->col_idx[M_mat->nnz] = T_mat->col_idx[i];
        mpfr_set(M_mat->values[M_mat->nnz], scaled_value, rnd);
        M_mat->nnz++;
    }
    
    mpfr_clear(scaled_value);
}

void build_matrices_optimized(SparseMatrix *R_mat, SparseMatrix *Q_mat, SparseMatrix *T_mat,
                            long Nmax, R_Coeffs *r_coeffs, Q_Coeffs *q_coeffs, T_Coeffs *t_coeffs,
                            mpfr_rnd_t rnd) {
    // Calculate exact number of non-zeros for each matrix
    long R_nnz = Nmax + 2 * (Nmax - 2);  // Main diagonal + 2 off-diagonals
    long Q_nnz = 2 * (Nmax - 1) + 2 * (Nmax - 3);  // 4 off-diagonals
    long T_nnz = Nmax + 2 * (Nmax - 2) + 2 * (Nmax - 4);  // Main diagonal + 4 off-diagonals
    
    // Initialize matrices with exact capacity
    init_sparse_matrix(R_mat, Nmax, Nmax, R_nnz);
    init_sparse_matrix(Q_mat, Nmax, Nmax, Q_nnz);
    init_sparse_matrix(T_mat, Nmax, Nmax, T_nnz);
    
    mpfr_t value;
    mpfr_init2(value, PREC);
    
    // Build R matrix (diagonals: -2, 0, +2)
    for (long n = 0; n < Nmax; n++) {
        // Diagonal (Element R(n,n))
        mpfr_neg(value, r_coeffs->R_n0[n], rnd);
        if (!mpfr_zero_p(value)) {
            R_mat->row_idx[R_mat->nnz] = n;
            R_mat->col_idx[R_mat->nnz] = n;
            mpfr_set(R_mat->values[R_mat->nnz], value, rnd);
            R_mat->nnz++;
        }
        
        // Lower Diagonal (Element R(n, n-2))
        if (n >= 2) {
            mpfr_set(value, r_coeffs->R_np2[n-2], rnd);
            if (!mpfr_zero_p(value)) {
                R_mat->row_idx[R_mat->nnz] = n;
                R_mat->col_idx[R_mat->nnz] = n-2;
                mpfr_set(R_mat->values[R_mat->nnz], value, rnd);
                R_mat->nnz++;
            }
        }
        
        // Upper Diagonal (Element R(n, n+2))
        if (n + 2 < Nmax) {
            mpfr_set(value, r_coeffs->R_nm2[n+2], rnd);
            if (!mpfr_zero_p(value)) {
                R_mat->row_idx[R_mat->nnz] = n;
                R_mat->col_idx[R_mat->nnz] = n+2;
                mpfr_set(R_mat->values[R_mat->nnz], value, rnd);
                R_mat->nnz++;
            }
        }
    }
    
    // Build Q matrix (diagonals: -3, -1, +1, +3)
    for (long n = 0; n < Nmax; n++) {
        // Lower Diagonals
        if (n >= 3) { // Element Q(n, n-3)
            mpfr_set(value, q_coeffs->Q_np3[n-3], rnd);
            if (!mpfr_zero_p(value)) {
                Q_mat->row_idx[Q_mat->nnz] = n;
                Q_mat->col_idx[Q_mat->nnz] = n-3;
                mpfr_set(Q_mat->values[Q_mat->nnz], value, rnd);
                Q_mat->nnz++;
            }
        }
        
        if (n >= 1) { // Element Q(n, n-1)
            mpfr_neg(value, q_coeffs->Q_np1[n-1], rnd);
            if (!mpfr_zero_p(value)) {
                Q_mat->row_idx[Q_mat->nnz] = n;
                Q_mat->col_idx[Q_mat->nnz] = n-1;
                mpfr_set(Q_mat->values[Q_mat->nnz], value, rnd);
                Q_mat->nnz++;
            }
        }
        
        // Upper Diagonals
        if (n + 1 < Nmax) { // Element Q(n, n+1)
            mpfr_set(value, q_coeffs->Q_nm1[n+1], rnd);
            if (!mpfr_zero_p(value)) {
                Q_mat->row_idx[Q_mat->nnz] = n;
                Q_mat->col_idx[Q_mat->nnz] = n+1;
                mpfr_set(Q_mat->values[Q_mat->nnz], value, rnd);
                Q_mat->nnz++;
            }
        }
        
        if (n + 3 < Nmax) { // Element Q(n, n+3)
            mpfr_neg(value, q_coeffs->Q_nm3[n+3], rnd);
            if (!mpfr_zero_p(value)) {
                Q_mat->row_idx[Q_mat->nnz] = n;
                Q_mat->col_idx[Q_mat->nnz] = n+3;
                mpfr_set(Q_mat->values[Q_mat->nnz], value, rnd);
                Q_mat->nnz++;
            }
        }
    }
    
    // Build T matrix (diagonals: -4, -2, 0, +2, +4)
    for (long n = 0; n < Nmax; n++) {
        // Lower Diagonals
        if (n >= 4) { // Element T(n, n-4)
            mpfr_set(value, t_coeffs->T_np4[n-4], rnd);
            if (!mpfr_zero_p(value)) {
                T_mat->row_idx[T_mat->nnz] = n;
                T_mat->col_idx[T_mat->nnz] = n-4;
                mpfr_set(T_mat->values[T_mat->nnz], value, rnd);
                T_mat->nnz++;
            }
        }
        
        if (n >= 2) { // Element T(n, n-2)
            mpfr_neg(value, t_coeffs->T_np2[n-2], rnd);
            if (!mpfr_zero_p(value)) {
                T_mat->row_idx[T_mat->nnz] = n;
                T_mat->col_idx[T_mat->nnz] = n-2;
                mpfr_set(T_mat->values[T_mat->nnz], value, rnd);
                T_mat->nnz++;
            }
        }
        
        // Diagonal (Element T(n,n))
        mpfr_set(value, t_coeffs->T_n0[n], rnd);
        if (!mpfr_zero_p(value)) {
            T_mat->row_idx[T_mat->nnz] = n;
            T_mat->col_idx[T_mat->nnz] = n;
            mpfr_set(T_mat->values[T_mat->nnz], value, rnd);
            T_mat->nnz++;
        }
        
        // Upper Diagonals
        if (n + 2 < Nmax) { // Element T(n, n+2)
            mpfr_neg(value, t_coeffs->T_nm2[n+2], rnd);
            if (!mpfr_zero_p(value)) {
                T_mat->row_idx[T_mat->nnz] = n;
                T_mat->col_idx[T_mat->nnz] = n+2;
                mpfr_set(T_mat->values[T_mat->nnz], value, rnd);
                T_mat->nnz++;
            }
        }
        
        if (n + 4 < Nmax) { // Element T(n, n+4)
            mpfr_set(value, t_coeffs->T_nm4[n+4], rnd);
            if (!mpfr_zero_p(value)) {
                T_mat->row_idx[T_mat->nnz] = n;
                T_mat->col_idx[T_mat->nnz] = n+4;
                mpfr_set(T_mat->values[T_mat->nnz], value, rnd);
                T_mat->nnz++;
            }
        }
    }
    
    mpfr_clear(value);
}

void build_K_matrix_optimized(SparseMatrix *K, SparseMatrix *M, SparseMatrix *R, mpfr_t sigma, long Nmax, mpfr_rnd_t rnd) {
    printf("Building K = M - sigma*R matrix (optimized)...\n");
    
    // Calculate exact size: K will have at most the sum of non-zeros from M and R
    // (worst case if there's no overlap)
    long K_max_nnz = M->nnz + R->nnz;
    
    // Initialize K with the exact capacity
    init_sparse_matrix(K, Nmax, Nmax, K_max_nnz);
    
    // Temporary variables
    mpfr_t neg_sigma_R_val;
    mpfr_init2(neg_sigma_R_val, PREC);
    
    // Create a temporary array to mark which positions are already filled in K
    // This is more efficient than searching through the arrays for each element
    char **filled = (char **)malloc(Nmax * sizeof(char *));
    for (long i = 0; i < Nmax; i++) {
        filled[i] = (char *)calloc(Nmax, sizeof(char)); // Initialize all to 0
    }
    
    // First, copy all elements from M to K
    for (long i = 0; i < M->nnz; i++) {
        long row = M->row_idx[i];
        long col = M->col_idx[i];
        
        K->row_idx[K->nnz] = row;
        K->col_idx[K->nnz] = col;
        mpfr_set(K->values[K->nnz], M->values[i], rnd);
        K->nnz++;
        
        // Mark this position as filled
        filled[row][col] = 1;
    }
    
    // Now process the R matrix elements (scaled by -sigma)
    for (long i = 0; i < R->nnz; i++) {
        long row = R->row_idx[i];
        long col = R->col_idx[i];
        
        // Calculate -sigma * R
        mpfr_mul(neg_sigma_R_val, R->values[i], sigma, rnd);
        mpfr_neg(neg_sigma_R_val, neg_sigma_R_val, rnd);
        
        if (filled[row][col]) {
            // Position already exists in K (from M), find it and add the value
            for (long j = 0; j < K->nnz; j++) {
                if (K->row_idx[j] == row && K->col_idx[j] == col) {
                    mpfr_add(K->values[j], K->values[j], neg_sigma_R_val, rnd);
                    break;
                }
            }
        } else {
            // Position doesn't exist in K yet, add it
            K->row_idx[K->nnz] = row;
            K->col_idx[K->nnz] = col;
            mpfr_set(K->values[K->nnz], neg_sigma_R_val, rnd);
            K->nnz++;
            
            // Mark as filled
            filled[row][col] = 1;
        }
    }
    
    // Clean up
    mpfr_clear(neg_sigma_R_val);
    for (long i = 0; i < Nmax; i++) {
        free(filled[i]);
    }
    free(filled);
    
    printf("Finished building K matrix. NNZ count in K: %ld\n", K->nnz);
}

void build_K_matrix_with_progress(SparseMatrix *K, SparseMatrix *M, SparseMatrix *R, mpfr_t sigma, long Nmax, mpfr_rnd_t rnd) {
    printf("Building K = M - sigma*R matrix (with progress updates)...\n");
    
    // Calculate total elements to process
    long total_elements = M->nnz + R->nnz;
    long processed_elements = 0;
    long next_progress_check = 0;
    int progress_percentage = 0;
    
    // Calculate exact size
    long K_max_nnz = total_elements;
    
    // Initialize K with the exact capacity
    init_sparse_matrix(K, Nmax, Nmax, K_max_nnz);
    
    // Temporary variables
    mpfr_t neg_sigma_R_val;
    mpfr_init2(neg_sigma_R_val, PREC);
    
    // Create a temporary array to mark filled positions
    char **filled = (char **)malloc(Nmax * sizeof(char *));
    for (long i = 0; i < Nmax; i++) {
        filled[i] = (char *)calloc(Nmax, sizeof(char));
    }
    
    clock_t start_time = clock();
    
    // First, copy all elements from M to K
    for (long i = 0; i < M->nnz; i++) {
        long row = M->row_idx[i];
        long col = M->col_idx[i];
        
        K->row_idx[K->nnz] = row;
        K->col_idx[K->nnz] = col;
        mpfr_set(K->values[K->nnz], M->values[i], rnd);
        K->nnz++;
        
        // Mark this position as filled
        filled[row][col] = 1;
        
        // Update progress
        processed_elements++;
        
        // Check if we should update progress display (every 5%)
        if (processed_elements >= next_progress_check) {
            progress_percentage = (int)(100.0 * processed_elements / total_elements);
            printf("Progress: %d%% complete                           \r", progress_percentage);
            fflush(stdout); // Ensure the output is displayed immediately
            next_progress_check = (progress_percentage + 1) * total_elements / 100;
        }
    }
    
    // Now process the R matrix elements (scaled by -sigma)
    for (long i = 0; i < R->nnz; i++) {
        long row = R->row_idx[i];
        long col = R->col_idx[i];
        
        // Calculate -sigma * R
        mpfr_mul(neg_sigma_R_val, R->values[i], sigma, rnd);
        mpfr_neg(neg_sigma_R_val, neg_sigma_R_val, rnd);
        
        if (filled[row][col]) {
            // Position already exists in K (from M), find it and add the value
            for (long j = 0; j < K->nnz; j++) {
                if (K->row_idx[j] == row && K->col_idx[j] == col) {
                    mpfr_add(K->values[j], K->values[j], neg_sigma_R_val, rnd);
                    break;
                }
            }
        } else {
            // Position doesn't exist in K yet, add it
            K->row_idx[K->nnz] = row;
            K->col_idx[K->nnz] = col;
            mpfr_set(K->values[K->nnz], neg_sigma_R_val, rnd);
            K->nnz++;
            
            // Mark as filled
            filled[row][col] = 1;
        }
        
        // Update progress
        processed_elements++;
        
        // Check if we should update progress display (every 1%)
        if (processed_elements >= next_progress_check) {
            progress_percentage = (int)(100.0 * processed_elements / total_elements);
            printf("Progress: %d%% complete                           \r", progress_percentage);
            fflush(stdout); // Ensure the output is displayed immediately
            next_progress_check = (progress_percentage + 1) * total_elements / 100;
        }
    }
    
    // Final newline after progress updates
    printf("\n");

    // Clean up
    mpfr_clear(neg_sigma_R_val);
    for (long i = 0; i < Nmax; i++) {
        free(filled[i]);
    }
    free(filled);
    
    clock_t end_time = clock();
    double actual_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("Finished building K matrix in %.2f seconds. NNZ count in K: %ld\n", 
           actual_time, K->nnz);
}

// Main function
int main(int argc, char **argv) {
    // Initialize parameters
    mpfr_t k_value, mu, k2, target_sigma, eigenvalue;
    long Nmax;
    mpfr_rnd_t rnd = MPFR_RNDN;

    mpfr_init2(k_value, PREC);
    mpfr_init2(mu, PREC);
    mpfr_init2(k2, PREC);
    mpfr_init2(target_sigma, PREC);
    mpfr_init2(eigenvalue, PREC);

    // Parse command line arguments (or use defaults)
    parse_args(argc, argv, k_value, mu, target_sigma, &Nmax, rnd);

    // Display parameters
    printf("Parameters:\n");
    mpfr_printf("  k_value = %.6Rg\n", k_value);
    mpfr_printf("  mu = %.6Rg\n", mu);
    mpfr_printf("  Target Sigma = %.10Rg\n", target_sigma);
    printf("  Nmax = %ld\n", Nmax);

    // Calculate k²
    mpfr_sqr(k2, k_value, rnd);

    clock_t start_time = clock();

    // Calculate coefficients
    printf("Initializing & Computing coefficients...\n");
    S1_Coeffs S1; init_S1(&S1, Nmax); compute_S1(&S1, Nmax, rnd);
    S2_Coeffs S2; init_S2(&S2, Nmax); compute_S2(&S2, &S1, Nmax, rnd);
    S3_Coeffs S3; init_S3(&S3, Nmax); compute_S3(&S3, &S2, &S1, Nmax, rnd);
    R_Coeffs R_coeffs; init_R(&R_coeffs, Nmax); compute_R(&R_coeffs, &S2, Nmax, k2, rnd);
    Q_Coeffs Q_coeffs; init_Q(&Q_coeffs, Nmax); compute_Q(&Q_coeffs, &S3, &S1, Nmax, k2, rnd);
    T_Coeffs T_coeffs; init_T(&T_coeffs, Nmax); compute_T(&T_coeffs, &S2, &R_coeffs, Nmax, k2, rnd);
    printf("Coefficients computed.\n");

    // Build the matrices
    printf("Building matrices...\n");
    SparseMatrix R_mat, Q_mat, T_mat, M_mat;
    
    // Initialize with estimated capacity
    init_sparse_matrix(&R_mat, Nmax, Nmax, 3*Nmax + 10);
    init_sparse_matrix(&Q_mat, Nmax, Nmax, 4*Nmax + 10);
    init_sparse_matrix(&T_mat, Nmax, Nmax, 5*Nmax + 10);

    // Build matrices from coefficients
    // build_matrices(&R_mat, &Q_mat, &T_mat, Nmax, &R_coeffs, &Q_coeffs, &T_coeffs, rnd);
    build_matrices_optimized(&R_mat, &Q_mat, &T_mat, Nmax, &R_coeffs, &Q_coeffs, &T_coeffs, rnd);
    printf("R, Q, T matrices built.\n");

    // Create M = Q + mu*T
    printf("Creating combined matrix M = Q + mu*T...\n");
    create_M_matrix_optimized(&M_mat, &Q_mat, &T_mat, mu, rnd);
    printf("M matrix created. NNZ=%ld\n", M_mat.nnz);

    // Allocate eigenvector
    mpfr_t *eigenvector = (mpfr_t *)malloc(Nmax * sizeof(mpfr_t));
    if (!eigenvector) {
        fprintf(stderr, "Failed to allocate eigenvector\n");
        exit(EXIT_FAILURE);
    }
    
    for (long i = 0; i < Nmax; i++) {
        mpfr_init2(eigenvector[i], PREC);
    }

    // Solve eigenvalue problem
    printf("Solving eigenvalue problem...\n");
    find_eigenvalue(&M_mat, &R_mat, &eigenvalue, eigenvector, Nmax, target_sigma, mu, rnd);

    // Print results
    printf("\nResults:\n");
    
    if (mpfr_nan_p(eigenvalue)) {
        printf("  Eigenvalue computation failed. Try adjusting parameters or increasing precision.\n");
    } else {
        mpfr_printf("  Eigenvalue = %.15RNf\n", eigenvalue);

        // Calculate Delta/mu
        mpfr_t delta, delta_over_mu, landau_ev_mpfr;
        mpfr_init2(delta, PREC);
        mpfr_init2(delta_over_mu, PREC);
        mpfr_init2(landau_ev_mpfr, PREC);
        mpfr_set_str(landau_ev_mpfr, LANDAU_EIGENVALUE, 10, rnd);

        mpfr_sub(delta, eigenvalue, landau_ev_mpfr, rnd);
        if (!mpfr_zero_p(mu)) {
            mpfr_div(delta_over_mu, delta, mu, rnd);
            mpfr_printf("  Delta/mu = %.6Rg\n", delta_over_mu);
        } else {
            printf("  Delta/mu = NaN (mu is zero)\n");
        }
        
        // Save eigenvalue and eigenvector to file
        save_eigen_data(eigenvalue, eigenvector, Nmax, k_value, mu, PREC);

        mpfr_clear(delta);
        mpfr_clear(delta_over_mu);
        mpfr_clear(landau_ev_mpfr);
    }

    // Print computation time
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Computation time: %.2f seconds\n", elapsed_time);

    // Clean up resources
    printf("Cleaning up main resources...\n");
    for (long i = 0; i < Nmax; i++) {
        mpfr_clear(eigenvector[i]);
    }
    free(eigenvector);

    free_sparse_matrix(&R_mat);
    free_sparse_matrix(&Q_mat);
    free_sparse_matrix(&T_mat);
    free_sparse_matrix(&M_mat);
    
    free_S1(&S1, Nmax);
    free_S2(&S2, Nmax);
    free_S3(&S3, Nmax);
    free_R(&R_coeffs, Nmax);
    free_Q(&Q_coeffs, Nmax);
    free_T(&T_coeffs, Nmax);
    
    mpfr_clear(k_value);
    mpfr_clear(mu);
    mpfr_clear(k2);
    mpfr_clear(target_sigma);
    mpfr_clear(eigenvalue);

    printf("Cleanup complete. Exiting.\n");
    return 0;
}
