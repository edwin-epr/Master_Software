/* ************************** */
/*      Utility Routines      */
/*     Numerical  Recipes     */
/* ************************** */
// -------------------------------------------------------------------------------------------------------------------
/* standard error handler */
void numericalError(char error_text[]);
// -------------------------------------------------------------------------------------------------------------------
/* allocates an float vector with subscript range v[nl..nh] */
int *intvector(long numberLower, long numberHigher);
// -------------------------------------------------------------------------------------------------------------------
/* free an int vector allocated with intvector() */
void free_intvector(int *vector, long numberLower, long numberHigher);
// -------------------------------------------------------------------------------------------------------------------
/* allocates an double vector with subscript range v[nl..nh] */
double *doublevector(long numberLower, long numberHigher);
// -------------------------------------------------------------------------------------------------------------------
/* free an double vector allocated with doublevector() */
void free_doublevector(double *vector, long numberLower, long numberHigher);
// -------------------------------------------------------------------------------------------------------------------
/* allocates a double matrix with range m[nrl..nrh][ncl..nch] */
double **doublematrix(long numberRowsLower, long numberRowsHigher, long numberColumnsLower, long numberColumnHigher);
// -------------------------------------------------------------------------------------------------------------------
/* free a double matrix allocated with doublematrix() */
void free_doublematrix(double **matrix, long numberRowsLower, long numberRowsHigher, long numberColumnsLower, long numberColumnsHigher);
// -------------------------------------------------------------------------------------------------------------------
/* ******************************* */
/* Functions for build test linear */
/* equation system                 */
/* ******************************* */
/* computes A matrix */
void system_matrix(double **matrix, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* computes b vector */
void terms_independents_vector(double *vector, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* *************************************** */
/*    Numerical linear algebra functions   */
/* *************************************** */
// -------------------------------------------------------------------------------------------------------------------
/* LU decomposition of a row wise permutation */
void ludcmp(double **a, int dimension, int *indx, double *d);
// -------------------------------------------------------------------------------------------------------------------
/* LU decomposition and backward substitution */
void lubksb(double **a, int dimension, int *indx, double b[]);
// -------------------------------------------------------------------------------------------------------------------
/* solve the linear set of equations Ax=b */
void solve(double **a, double *b, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* compute inverse of a matrix */
void inverse(double **a, double **y, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* compute inverse of a diagonal matrix */
void inverse_diagonal_matrix(double **inverseMatrix, double **matrix, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* compute 2-norm of a vector */
double vector_norm_2(double *vector, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* compute 2-norm of a matrix */
double matrix_norm_2(double **matrix, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* copy a vectors to other vector */
/* y <- x */
void vector_copy(double *vectorY, double *vectorX, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* compute dot product of two vectors */
/* 〈x, y〉  */
double vector_dot(double *vectorX, double *vectorY, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* computes vector addition */
/* z <- y + alpha*x */
void vector_add(double *vectorZ, double *vectorY, double alpha, double *vectorX, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* computes scalar-vector multiplication */
/* y <- alpha*x */
void vector_scalar_multiplication(double *vectorY, double alpha, double *vectorX, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* computes matrix-vector multiplication */
/* y <- alpha*A*x */
void matrix_vector_multiplication(double *vectorY, double alpha, double **matrix, double *vectorX, int row, int column);
// -------------------------------------------------------------------------------------------------------------------

/* computes scalar-matrix multiplication */
/* B <- alpha*A */
void matrix_scalar_multiplication(double **matrixB, double alpha, double **matrixA, int dimension);
// -------------------------------------------------------------------------------------------------------------------

/* computes matrix addition */
/* C <- B + alpha*A */
void matrix_add(double **matrixC, double **matrixB, double alpha, double **matrixA, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* computes error of xs_method approximation */
double method_error(double *exactSolutionVector, double *approximateSolutionVector, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* Computes A = D+L+U split */
/* obtain the diagonal of a matrix */
/* D <- diag(A) */
void diagonal_matrix(double **matrixD, double **matrixA, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* obtain the portion lower triangular strict of a matrix */
/* L <- lower_triangular(A) */
void lower_triangular_matrix(double **matrixL, double **matrixA, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* obtain the portion upper triangular strict of a matrix */
/* U <- upper_triangular(A) */
void upper_triangular_matrix(double **matrixU, double **matrixA, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* Functions for GMRES Method */
/* get the ith column of a matrix and put in a vector */
void matrix_to_vector(double **matrix, double *vector, int columnIndex, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* put the elements of a vector in the ith column of a matrix */
void vector_to_matrix(double *vector, double **matrix, int columnIndex, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* compute x tal que U*x = b, where U is a upper triangular matrix */
void backward_substitution(double **matrixU, double *vectorB, double *vectorX, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* Output routines */
/* print a int vector */
void print_intvector(int *vector, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* print a double vector */
void print_doublevector(double *vector, int dimension);
// -------------------------------------------------------------------------------------------------------------------
/* print a double matrix */
void print_doublematrix(double **matrix, int rows, int columns);
// -------------------------------------------------------------------------------------------------------------------
/* print the relative error, run time and entirely iterations of methods */
void method_information(double methodError, double runTime, int maximumIteration);
// -------------------------------------------------------------------------------------------------------------------
double minimum_value(double *vector, int vectorDimension);
