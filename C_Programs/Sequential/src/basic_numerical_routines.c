#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// -------------------------------------------------------------------------------------------------------------------
#include "basic_numerical_routines.h"
// -------------------------------------------------------------------------------------------------------------------
#define NR_END 1
#define FREE_ARG char *
#define TINY 1.0e-20
#define MINUS -1.0
#define PLUS 1.0
#define THRESHOLD 1.0e-16
// -------------------------------------------------------------------------------------------------------------------
// Numerical Recipes
// Function definitions
// -------------------------------------------------------------------------------------------------------------------
void numericalError(char error_text[])
// Numerical Recipes standard error handler
{
	fprintf(stderr, "Numerical Recipes run-time error...\n");
	fprintf(stderr, "%s\n", error_text);
	fprintf(stderr, "...now exiting to system...\n");
	exit(1);
}
// -------------------------------------------------------------------------------------------------------------------
int *intvector(long numberLower, long numberHigher)
// allocates an int vector with subscript range vector[numberLower..numberHigher]
{
	int *vector;
	vector = (int *)malloc((size_t)((numberHigher - numberLower + 1 + NR_END) * sizeof(int)));
	if (!vector)
		numericalError("allocation failure in intVector()");
	return vector - numberLower + NR_END;
}
// -------------------------------------------------------------------------------------------------------------------
void free_intvector(int *vector, long numberLower, long numberHigher)
// free an int vector allocated with intvector()
{
	free((FREE_ARG)(vector + numberLower - NR_END));
}
// -------------------------------------------------------------------------------------------------------------------
double *doublevector(long numberLower, long numberHigher)
// allocates an double vector with subscript range vector[numberLower..numberHigher]
{
	double *vector;
	vector = (double *)malloc((size_t)((numberHigher - numberLower + 1 + NR_END) * sizeof(double)));
	if (!vector)
		numericalError("allocation failure in doublevector()");
	return vector - numberLower + NR_END;
}
// -------------------------------------------------------------------------------------------------------------------
void free_doublevector(double *vector, long numberLower, long numberHigher)
// free an double vector allocated with doublevector()
{
	free((FREE_ARG)(vector + numberLower - NR_END));
}
// -------------------------------------------------------------------------------------------------------------------
double **doublematrix(long numberRowsLower, long numberRowsHigher, long numberColumnsLower, long numberColumnsHigher)
/* allocates an double matrix with subscript range matrix[numberRowLower..numberRowHigher][numberColumnsLower..numberColumnsHigher] */
{
	long i, numberRows = numberRowsHigher - numberRowsLower + 1, numberColumns = numberColumnsHigher - numberColumnsLower + 1;
	double **matrix;
	/* allocates pointers to rows */
	matrix = (double **)malloc((size_t)((numberRows + NR_END) * sizeof(double *)));
	if (!matrix)
		numericalError("allocation failure 1 in matrix()");
	matrix += NR_END;
	matrix -= numberRowsLower;
	/* allocate rows and set pointers to them */
	matrix[numberRowsLower] = (double *)malloc((size_t)((numberRows * numberColumns + NR_END) * sizeof(double)));
	if (!matrix[numberRowsLower])
		numericalError("allocation failure 2 in matrix()");
	matrix[numberRowsLower] += NR_END;
	matrix[numberRowsLower] -= numberColumnsLower;
	for (i = numberRowsLower + 1; i <= numberRowsHigher; i++)
		matrix[i] = matrix[i - 1] + numberColumns;
	/* return pointer to array of pointers to rows */
	return matrix;
}
// -------------------------------------------------------------------------------------------------------------------
void free_doublematrix(double **matrix, long numberRowsLower, long numberRowsHigher, long numberColumnsLower, long numberColumnsHigher)
/* free an double matrix allocated with doublematrix() */
{
	free((FREE_ARG)(matrix[numberRowsLower] + numberColumnsLower - NR_END));
	free((FREE_ARG)(matrix + numberRowsLower - NR_END));
}
// -------------------------------------------------------------------------------------------------------------------
/* Build Linear Equation System of test */
void system_matrix(double **matrix, int dimension)
// computes A matrix of Ax=b
{
	for (int i = 1; i <= dimension; i++)
	{
		for (int j = 1; j <= dimension; j++)
		{
			if (i == j)
			{
				matrix[i][j] = 3.0;

				if (i >= 2)
				{
					matrix[i - 1][j] = -1.0;
				}
				if (j >= 2)
				{
					matrix[i][j - 1] = -1.0;
				}
			}
			else
			{
				matrix[i][j] = 0.0;
			}
			if (j == (dimension - i + 1) && matrix[i][j] == 0)
			{
				matrix[i][j] = 0.5;
			}
		}
	}
}
// -------------------------------------------------------------------------------------------------------------------
void terms_independents_vector(double *vector, int dimension)
// computes b vector of Ax=b
{
	vector[1] = 2.5;
	vector[dimension] = 2.5;
	int n = (int)floor(dimension / 2);
	for (int i = 2; i <= dimension - 1; i++)
	{
		if (i == n || i == n + 1)
		{
			vector[i] = 1.0;
		}
		else
		{
			vector[i] = 1.5;
		}
	}
}
// -------------------------------------------------------------------------------------------------------------------
/* Matrices and vectors operations */
/* LU Decomposition routines */
// Given a matrix a[1...dim][1...dim], this routine replaces it by the LU
// decomposition of a row wise permutation of itself
void ludcmp(double **a, int dim, int *indx, double *d)
{
	int i, imax, j, k;
	double big, dum, sum, temp;
	double *vv;
	vv = doublevector(1, dim);
	*d = 1.0;
	for (i = 1; i <= dim; i++)
	{
		big = 0.0;
		for (j = 1; j <= dim; j++)
			if ((temp = fabs(a[i][j])) > big)
				big = temp;
		if (big == 0.0)
			numericalError("Singular matrix in routine ludcmp");
		vv[i] = 1.0 / big;
	}
	for (j = 1; j <= dim; j++)
	{
		for (i = 1; i < j; i++)
		{
			sum = a[i][j];
			for (k = 1; k < i; k++)
				sum -= a[i][k] * a[k][j];
			a[i][j] = sum;
		}
		big = 0.0;
		for (i = j; i <= dim; i++)
		{
			sum = a[i][j];
			for (k = 1; k < j; k++)
				sum -= a[i][k] * a[k][j];
			a[i][j] = sum;
			if ((dum = vv[i] * fabs(sum)) >= big)
			{
				big = dum;
				imax = i;
			}
		}
		if (j != imax)
		{
			for (k = 1; k <= dim; k++)
			{
				dum = a[imax][k];
				a[imax][k] = a[j][k];
				a[j][k] = dum;
			}
			*d = -(*d);
			vv[imax] = vv[j];
		}
		indx[j] = imax;
		if (a[j][j] == 0.0)
			a[j][j] = TINY;
		if (j != dim)
		{
			dum = 1.0 / (a[j][j]);
			for (i = j + 1; i <= dim; i++)
				a[i][j] *= dum;
		}
	}
	free_doublevector(vv, 1, dim);
}
// -------------------------------------------------------------------------------------------------------------------
// Solves the set of dim linear equation Ax=B. Here a[1...dim][1...dim] is
// input, no as the matrix A but rather a is LU decomposition, determined
// by the routine ludcmp
void lubksb(double **a, int dim, int *indx, double b[])
{
	int ii = 0, ip, j;
	double sum;

	for (int i = 1; i <= dim; i++)
	{
		ip = indx[i];
		sum = b[ip];
		b[ip] = b[i];
		if (ii)
			for (j = ii; j <= i - 1; j++)
				sum -= a[i][j] * b[j];
		else if (sum)
			ii = i;
		b[i] = sum;
	}
	for (int i = dim; i >= 1; i--)
	{
		sum = b[i];
		for (j = i + 1; j <= dim; j++)
			sum -= a[i][j] * b[j];
		b[i] = sum / a[i][i];
	}
}
// -------------------------------------------------------------------------------------------------------------------
/* Solves using lu decomposition and backward substitution */
/* solve the set of n linear equation Ax=b */
void solve(double **a, double *b, int dim)
{
	double d;
	int *indx;
	indx = intvector(1, dim);
	ludcmp(a, dim, indx, &d);
	lubksb(a, dim, indx, b);
	// print_doublevector(b, dim);
}
// -------------------------------------------------------------------------------------------------------------------
void inverse(double **a, double **y, int dim)
/* Finds the inverse of a matrix column by column */
/* matrix y will now contain the inverse of inverse of the */
/* original matrix a which will have been destroyed */
{
	double d, *col;
	int *indx;
	indx = intvector(1, dim);
	/* lu decomposition */
	ludcmp(a, dim, indx, &d);
	/* matrix inversion */
	col = doublevector(1, dim);
	for (int j = 1; j <= dim; j++)
	{
		for (int i = 1; i <= dim; i++)
			col[i] = 0.0;
		col[j] = 1.0;
		lubksb(a, dim, indx, col);
		for (int i = 1; i <= dim; i++)
			y[i][j] = col[i];
	}
}
// -------------------------------------------------------------------------------------------------------------------
/* compute the inverse of a diagonal matrix */
void inverse_diagonal_matrix(double **inverseMatrix, double **matrix, int dimension)
/* computes D matrix as a diagonal of A */
{
	for (int i = 1; i <= dimension; i++)
	{
		inverseMatrix[i][i] = 1 / matrix[i][i];
	}
}
// -------------------------------------------------------------------------------------------------------------------
/* computes the 2-norm of a vector */
double vector_norm_2(double *vector, int dimension)
{
	double product;
	double sum = 0, norm;
	for (int i = 1; i <= dimension; i++)
	{
		product = vector[i] * vector[i];
		sum += product;
	}
	norm = sqrt(sum);
	return norm;
}
// -------------------------------------------------------------------------------------------------------------------
/* computes the 2-norm of a matrix */
double matrix_norm_2(double **matrix, int dimension)
{
	double product, norm;
	double sum = 0;
	for (int i = 1; i <= dimension; i++)
	{
		for (int j = 1; j <= dimension; j++)
		{
			product = matrix[i][j] * matrix[i][j];
			sum += product;
		}
	}
	norm = sqrt(sum);
	return norm;
}
// -------------------------------------------------------------------------------------------------------------------
/* copy a vectors to other vector */
/* y <- x */
void vector_copy(double *vectorY, double *vectorX, int dimension)
{
	for (int i = 1; i <= dimension; i++)
	{
		vectorY[i] = vectorX[i];
	}
}
// -------------------------------------------------------------------------------------------------------------------
/* computes dot product of two vectors */
/* 〈x, y〉  */
double vector_dot(double *vectorX, double *vectorY, int dimension)
{
	double sum = 0.0;
	for (int i = 1; i <= dimension; i++)
	{
		sum += vectorX[i] * vectorY[i];
	}
	return sum;
}
// -------------------------------------------------------------------------------------------------------------------
/* computes vector addition */
/* z <- y + alpha*x */
void vector_add(double *vectorZ, double *vectorY, double alpha, double *vectorX, int dimension)
{
	for (int i = 1; i <= dimension; i++)
	{
		vectorZ[i] = vectorY[i] + alpha * vectorX[i];
	}
}
// -------------------------------------------------------------------------------------------------------------------
/* computes scalar-vector multiplication */
/* y <- alpha*x */
void vector_scalar_multiplication(double *vectorY, double alpha, double *vectorX, int dimension)
{
	for (int i = 1; i <= dimension; i++)
	{
		vectorY[i] = alpha * vectorX[i];
	}
}
// -------------------------------------------------------------------------------------------------------------------
/* computes matrix-vector multiplication */
/* y <- alpha*A*x */
void matrix_vector_multiplication(double *vectorY, double alpha, double **matrix, double *vectorX, int row, int column)
/* computes matrix-vector multiplication as y = alpha*A*x */
{
	for (int i = 1; i <= row; i++)
	{
		vectorY[i] = 0.0;
		for (int j = 1; j <= column; j++)
		{
			vectorY[i] += alpha * matrix[i][j] * vectorX[j];
		}
	}
}
// -------------------------------------------------------------------------------------------------------------------
/* computes scalar-matrix multiplication */
/* B <- alpha*A */
void matrix_scalar_multiplication(double **matrixB, double alpha, double **matrixA, int dimension)
{
	for (int i = 1; i <= dimension; i++)
	{
		for (int j = 1; j <= dimension; j++)
		{
			matrixB[i][j] = alpha * matrixA[i][j];
		}
	}
}
// -------------------------------------------------------------------------------------------------------------------
/* computes matrix addition */
/* C <- B + alpha*A */
void matrix_add(double **matrixC, double **matrixB, double alpha, double **matrixA, int dimension)
{
	for (int i = 1; i <= dimension; i++)
	{
		for (int j = 1; j <= dimension; j++)
		{
			matrixC[i][j] = matrixB[i][j] + alpha * matrixA[i][j];
		}
	}
}
// -------------------------------------------------------------------------------------------------------------------
/* computes error of xs_method approximation */
/* error_method = ||xs-xe|| / || xs || */
double method_error(double *exactSolutionVector, double *approximateSolutionVector, int dimension)
{
	double methodError = 0.0;
	double *vectorError = doublevector(1, dimension);
	vector_add(vectorError, approximateSolutionVector, MINUS, exactSolutionVector, dimension);
	methodError = vector_norm_2(vectorError, dimension) / vector_norm_2(approximateSolutionVector, dimension);
	free_doublevector(vectorError, 1, dimension);
	return methodError;
}
// -------------------------------------------------------------------------------------------------------------------
/* Computes A = D+L+U split */
/* obtain the diagonal of a matrix */
/* D <- diag(A) */
void diagonal_matrix(double **matrixD, double **matrixA, int dimension)
/* computes D matrix as a diagonal of A */
{
	for (int i = 1; i <= dimension; i++)
	{
		matrixD[i][i] = matrixA[i][i];
	}
}
// -------------------------------------------------------------------------------------------------------------------
/* obtain the portion lower triangular strict of a matrix */
/* L <- lower_triangular(A) */
void lower_triangular_matrix(double **matrixL, double **matrixA, int dimension)
{
	for (int i = 1; i <= dimension; i++)
	{
		for (int j = 1; j <= i - 1; j++)
		{
			matrixL[i][j] = matrixA[i][j];
		}
	}
}
// -------------------------------------------------------------------------------------------------------------------
/* obtain the portion upper triangular strict of a matrix */
/* U <- upper_triangular(A) */
void upper_triangular_matrix(double **matrixU, double **matrixA, int dimension)
{
	for (int i = 1; i <= dimension; i++)
	{
		for (int j = i + 1; j <= dimension; j++)
		{
			matrixU[i][j] = matrixA[i][j];
		}
	}
}
// -------------------------------------------------------------------------------------------------------------------
/* Functions for GMRES Method */
/* get the ith column of a matrix and put in a vector */
void matrix_to_vector(double **matrix, double *vector, int columnIndex, int dimension)
{
	for (int k = 1; k <= dimension; k++)
		vector[k] = matrix[k][columnIndex];
}
// -------------------------------------------------------------------------------------------------------------------
/* put the elements of a vector in the ith column of a matrix */
void vector_to_matrix(double *vector, double **matrix, int columnIndex, int dimension)
{
	for (int k = 1; k <= dimension; k++)
		matrix[k][columnIndex] = vector[k];
}
// -------------------------------------------------------------------------------------------------------------------
/* compute x tal que U*x = b, where U is a upper triangular matrix */
void backward_substitution(double **matrixU, double *vectorB, double *vectorX, int dimension)
{
	vectorX[dimension] = vectorB[dimension] / matrixU[dimension][dimension];
	for (int i = dimension - 1; i >= 1; i--)
	{
		double sum = 0.0;
		for (int j = i + 1; j <= dimension; j++)
			sum += matrixU[i][j] * vectorX[j];
		vectorX[i] = (vectorB[i] - sum) / matrixU[i][i];
	}
}
// -------------------------------------------------------------------------------------------------------------------
// Output routines
// -------------------------------------------------------------------------------------------------------------------
void print_intvector(int *vector, int dimension)
/* print a int vector */
{
	for (int i = 1; i <= dimension; ++i)
	{
		printf("%d ", vector[i]);
		printf("\n");
	}
	printf("\n");
}
// -------------------------------------------------------------------------------------------------------------------
void print_doublevector(double *vector, int dimension)
/* print a double vector */
{
	for (int i = 1; i <= dimension; ++i)
	{
		printf("%f ", vector[i]);
		printf("\n");
	}
	printf("\n");
}
// -------------------------------------------------------------------------------------------------------------------
void print_doublematrix(double **matrix, int rows, int columns)
/* print a double matrix */
{
	for (int i = 1; i <= rows; ++i)
	{
		for (int j = 1; j <= columns; ++j)
			printf("%f ", matrix[i][j]);
		printf("\n");
	}
	printf("\n");
}
// -------------------------------------------------------------------------------------------------------------------
void method_information(double methodError, double runTime, int maximumIteration)
/* print the relative error, run time and entirely iterations of methods */
{
	printf("error: %.4e, time: %.2e s, iterations: %d\n", methodError, runTime, maximumIteration);
}
// -------------------------------------------------------------------------------------------------------------------
double minimum_value(double *vector, int vectorDimension)
{
	double minimum = vector[1];
	for (int i = 2; i <= vectorDimension; i++)
	{
		if (minimum > vector[i])
		{
			minimum = vector[i];
		}
	}

	return minimum;
}
// -------------------------------------------------------------------------------------------------------------------