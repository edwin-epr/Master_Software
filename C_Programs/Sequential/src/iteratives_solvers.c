/* *********************************************************************** */
// 									Iterative Solvers
/* *********************************************************************** */
/*
 * barVectorX: vector solución exacta del sistema Ax = b
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "basic_numerical_routines.h"
#include "iteratives_solvers.h"

#define PLUS 1.0
#define MINUS -1.0
#define INT_ONE 1
#define DOUBLE_ONE 1.0
/*
									Jacobi Method
*/
double jacobi_method(
	double **matrixInverseD,
	double **matrixT,
	double *vectorB,
	double *barVectorX,
	double *vectorX_initial,
	int maximumIteration,
	int dimension)
{
	double methodError = 0.0;
	double *vectorX_current, *vectorX_next;
	double *vectorY, *vectorF;

	vectorX_current = doublevector(INT_ONE, dimension);
	vectorX_next = doublevector(INT_ONE, dimension);
	vectorY = doublevector(INT_ONE, dimension);
	vectorF = doublevector(INT_ONE, dimension);

	vector_copy(vectorX_current, vectorX_initial, dimension);

	for (int k = 1; k <= maximumIteration; k++)
	{
		/* computes (L+U)*x(k) vector */
		matrix_vector_multiplication(vectorY, DOUBLE_ONE, matrixT, vectorX_current, dimension, dimension);
		// sparseMatrixVectorMultiplication(vectorY, DOUBLE_ONE, sparseT, vectorX_current, dimension);

		/* computes b - (L+U)*x(k) vector */
		vector_add(vectorF, vectorB, MINUS, vectorY, dimension);

		/* computes x(k+1) = inv(D)*(b - (L+U)*x(k)) vector */
		matrix_vector_multiplication(vectorX_next, DOUBLE_ONE, matrixInverseD, vectorF, dimension, dimension);
		// sparseMatrixVectorMultiplication(vectorX_next, DOUBLE_ONE, sparseInverseD, vectorF, dimension);

		/* computes error method and stop criterion */
		if (k != maximumIteration)
		{
			/* xp_vec <-- xs_method */
			vector_copy(vectorX_current, vectorX_next, dimension);
			// break;
		}
		else
		{
			methodError = method_error(barVectorX, vectorX_next, dimension);
		}
	}

	free_doublevector(vectorF, INT_ONE, dimension);
	free_doublevector(vectorY, INT_ONE, dimension);
	free_doublevector(vectorX_next, INT_ONE, dimension);
	free_doublevector(vectorX_current, INT_ONE, dimension);

	return methodError;
}
/*
								Gauss-Seidel Method
*/
double gauss_seidel_method(
	double **inverseMatrixT,
	double **matrixU,
	double *vectorB,
	double *barVectorX,
	double *vectorX_initial,
	int maximumIteration,
	int dimension)
{
	double methodError = 0.0;
	double *vectorX_current, *vectorX_next;
	double *vectorY, *vectorF;

	vectorX_current = doublevector(INT_ONE, dimension);
	vectorX_next = doublevector(INT_ONE, dimension);
	vectorY = doublevector(INT_ONE, dimension);
	vectorF = doublevector(INT_ONE, dimension);

	vector_copy(vectorX_current, vectorX_initial, dimension);

	for (int k = 1; k <= maximumIteration; k++)
	{
		/* computes U*x(k) */
		matrix_vector_multiplication(vectorY, DOUBLE_ONE, matrixU, vectorX_current, dimension, dimension);

		/* computes b - U*x(k) */
		vector_add(vectorF, vectorB, MINUS, vectorY, dimension);

		/* computes x(k+1) = inv(D+L)*(b - U*x(k)) */
		matrix_vector_multiplication(vectorX_next, DOUBLE_ONE, inverseMatrixT, vectorF, dimension, dimension);

		/* computes error method and stop criterion */
		if (k != maximumIteration)
		{
			/* xp_vec <-- xs_method */
			vector_copy(vectorX_current, vectorX_next, dimension);
			// break;
		}
		else
		{
			methodError = method_error(barVectorX, vectorX_next, dimension);
		}
	}

	free_doublevector(vectorF, INT_ONE, dimension);
	free_doublevector(vectorY, INT_ONE, dimension);
	free_doublevector(vectorX_next, INT_ONE, dimension);
	free_doublevector(vectorX_current, INT_ONE, dimension);

	return methodError;
}
/*
						Successive Over-Relaxation Method
*/
double sor_method(
	double **inverseMatrixT,
	double **matrixOmegaD,
	double **matrixOmegaU,
	double *vectorF,
	double *barVectorX,
	double *vectorX_initial,
	int maximumIteration,
	int dimension)
{
	double methodError = 0.0;
	double *vectorX_current, *vectorX_next;
	double *vectorV, *vectorW, *vectorY, *vectorZ;

	vectorX_current = doublevector(INT_ONE, dimension);
	vectorX_next = doublevector(INT_ONE, dimension);
	vectorV = doublevector(INT_ONE, dimension);
	vectorW = doublevector(INT_ONE, dimension);
	vectorY = doublevector(INT_ONE, dimension);
	vectorZ = doublevector(INT_ONE, dimension);

	vector_copy(vectorX_current, vectorX_initial, dimension);

	for (int k = 1; k <= maximumIteration; k++)
	{
		/* computes (1-omega)*D*x(k) vector */
		matrix_vector_multiplication(vectorV, DOUBLE_ONE, matrixOmegaD, vectorX_current, dimension, dimension);

		/* computes omega*U*x(k) vector */
		matrix_vector_multiplication(vectorW, DOUBLE_ONE, matrixOmegaU, vectorX_current, dimension, dimension);

		/* computes (1-omega)*D*x(k) - omega*U*x(k) vector */
		vector_add(vectorY, vectorV, MINUS, vectorW, dimension);

		/* computes inv(D+omega*L)*(1-omega)*D*x(k) - omega*U*x(k) vector */
		matrix_vector_multiplication(vectorZ, DOUBLE_ONE, inverseMatrixT, vectorY, dimension, dimension);

		/* computes x(k+1) = inv(D+omega*L)*((1-omega)*D*x(k) - omega*U*x(k)) + omega*(D+omega*L)*b vector */
		vector_add(vectorX_next, vectorZ, PLUS, vectorF, dimension);

		/* computes error method and stop criterion */
		if (k != maximumIteration)
		{
			/* xp_vec <-- xs_method */
			vector_copy(vectorX_current, vectorX_next, dimension);
			// break;
		}
		else
		{
			methodError = method_error(barVectorX, vectorX_next, dimension);
		}
	}

	free_doublevector(vectorZ, INT_ONE, dimension);
	free_doublevector(vectorY, INT_ONE, dimension);
	free_doublevector(vectorW, INT_ONE, dimension);
	free_doublevector(vectorV, INT_ONE, dimension);
	free_doublevector(vectorX_next, INT_ONE, dimension);
	free_doublevector(vectorX_current, INT_ONE, dimension);

	return methodError;
}
double conjugate_gradient_method(
	double **matrixA,
	double *vectorB,
	double *barVectorX,
	double *vectorX_initial,
	int maximumIteration,
	int dimension)
{
	double alpha, beta, methodError = 0.0;
	double *vectorX_current, *vectorX_next;
	double *vectorR, *vectorSine, *vectorBetaE, *vectorVj;
	double *vectorY, *vectorZ;

	vectorX_current = doublevector(INT_ONE, dimension);
	vectorX_next = doublevector(INT_ONE, dimension);
	vectorR = doublevector(INT_ONE, dimension);
	vectorSine = doublevector(INT_ONE, dimension);
	vectorBetaE = doublevector(INT_ONE, dimension);
	vectorVj = doublevector(INT_ONE, dimension);
	vectorY = doublevector(INT_ONE, dimension);
	vectorZ = doublevector(INT_ONE, dimension);

	vector_copy(vectorX_current, vectorX_initial, dimension);

	matrix_vector_multiplication(vectorY, DOUBLE_ONE, matrixA, vectorX_current, dimension, dimension);

	vector_add(vectorR, vectorB, MINUS, vectorY, dimension);

	vector_copy(vectorBetaE, vectorR, dimension);

	for (int k = 1; k <= maximumIteration; k++)
	{
		/* Compute alpha(k) = r^T(k)r(k) / d^T(k)Ad(k) */
		matrix_vector_multiplication(vectorY, DOUBLE_ONE, matrixA, vectorBetaE, dimension, dimension);
		alpha = vector_dot(vectorR, vectorR, dimension) / vector_dot(vectorBetaE, vectorY, dimension);

		/* Compute x(k+1) = x(k) + alpha(k)*d(k) */
		vector_scalar_multiplication(vectorZ, alpha, vectorBetaE, dimension);
		vector_add(vectorX_next, vectorX_current, PLUS, vectorZ, dimension);

		/* computes error method and stop criterion */
		if (k != maximumIteration)
		{
			/* Compute r(k+1) = r(k) - alpha(k)*A*d(k) */
			vector_scalar_multiplication(vectorZ, alpha, vectorY, dimension);
			vector_add(vectorSine, vectorR, MINUS, vectorZ, dimension);

			/* Compute beta(k) = r^T(k+1)r(k+1) / r^T(k)r(k) */
			beta = vector_dot(vectorSine, vectorSine, dimension) / vector_dot(vectorR, vectorR, dimension);

			/* Compute d(k+1) = r(k+1) + beta(k)*d(k) */
			vector_scalar_multiplication(vectorZ, beta, vectorBetaE, dimension);
			vector_add(vectorVj, vectorSine, PLUS, vectorZ, dimension);

			/* xp_vec <-- xs_method */
			vector_copy(vectorR, vectorSine, dimension);
			vector_copy(vectorBetaE, vectorVj, dimension);
			vector_copy(vectorX_current, vectorX_next, dimension);
		}
		else
		{
			methodError = method_error(barVectorX, vectorX_next, dimension);
		}
	}

	free_doublevector(vectorZ, INT_ONE, dimension);
	free_doublevector(vectorY, INT_ONE, dimension);
	free_doublevector(vectorVj, INT_ONE, dimension);
	free_doublevector(vectorBetaE, INT_ONE, dimension);
	free_doublevector(vectorSine, INT_ONE, dimension);
	free_doublevector(vectorR, INT_ONE, dimension);
	free_doublevector(vectorX_next, INT_ONE, dimension);
	free_doublevector(vectorX_current, INT_ONE, dimension);

	return methodError;
}

double biconjugate_gradient_stabilized_method(
	double **matrixA,
	double *vectorB,
	double *barVectorX,
	double *vectorX_initial,
	int maximumIteration,
	int dimension)
{
	double alpha, beta, omega, methodError = 0.0;
	double *vectorX_current, *vectorX_next;
	double *vectorR, *vectorSine, *vectorCosine;
	double *vectorBetaE, *vectorVj, *vectorQj;
	double *vectorU, *vectorV, *vectorW, *vectorY, *vectorZ;

	vectorX_current = doublevector(INT_ONE, dimension);
	vectorX_next = doublevector(INT_ONE, dimension);
	vectorR = doublevector(INT_ONE, dimension);
	vectorSine = doublevector(INT_ONE, dimension);
	vectorCosine = doublevector(INT_ONE, dimension);
	vectorBetaE = doublevector(INT_ONE, dimension);
	vectorVj = doublevector(INT_ONE, dimension);
	vectorQj = doublevector(INT_ONE, dimension);
	vectorU = doublevector(INT_ONE, dimension);
	vectorV = doublevector(INT_ONE, dimension);
	vectorW = doublevector(INT_ONE, dimension);
	vectorY = doublevector(INT_ONE, dimension);
	vectorZ = doublevector(INT_ONE, dimension);

	vector_copy(vectorX_current, vectorX_initial, dimension);

	matrix_vector_multiplication(vectorU, DOUBLE_ONE, matrixA, vectorX_current, dimension, dimension);

	vector_add(vectorR, vectorB, MINUS, vectorU, dimension);

	vector_copy(vectorBetaE, vectorR, dimension);

	vector_copy(vectorCosine, vectorR, dimension);

	for (int k = 1; k <= maximumIteration; k++)
	{
		/* Compute alpha(k) = <r(k), r*(0)> / <r*(0), Ad(k)> */
		matrix_vector_multiplication(vectorU, DOUBLE_ONE, matrixA, vectorBetaE, dimension, dimension);
		alpha = vector_dot(vectorR, vectorCosine, dimension) / vector_dot(vectorCosine, vectorU, dimension);

		/* Compute s(k) = r(k) - alpha(k)*A*p(k) */
		vector_scalar_multiplication(vectorW, alpha, vectorU, dimension);
		vector_add(vectorQj, vectorR, MINUS, vectorW, dimension);

		/* Compute omega(k) = <A*s(k), s(k)> / <A*s(k), A*s(k)> */
		matrix_vector_multiplication(vectorV, DOUBLE_ONE, matrixA, vectorQj, dimension, dimension);
		omega = vector_dot(vectorV, vectorQj, dimension) / vector_dot(vectorV, vectorV, dimension);

		/* Compute x(k+1) = x(k) + alpha(k)*p(k) + omega(k)*s(k) */
		vector_scalar_multiplication(vectorY, alpha, vectorBetaE, dimension);
		vector_scalar_multiplication(vectorZ, omega, vectorQj, dimension);
		vector_add(vectorY, vectorY, PLUS, vectorZ, dimension);
		vector_add(vectorX_next, vectorX_current, PLUS, vectorY, dimension);

		/* computes error method and stop criterion */
		if (k != maximumIteration)
		{
			/* Compute r(k+1) = s(k) - omega(k)*A*s(k) */
			vector_scalar_multiplication(vectorW, omega, vectorV, dimension);
			vector_add(vectorSine, vectorQj, MINUS, vectorW, dimension);

			/* Compute beta(k) = <r(k+1), r*(0)>/<r(k), r*(0)> * alpha(k)/omega(k) */
			beta = vector_dot(vectorSine, vectorCosine, dimension) / vector_dot(vectorR, vectorCosine, dimension);

			/* Compute d(k+1) = r(k+1) + beta(k)*(d(k)-omega(k)*A*p(k)) */
			vector_scalar_multiplication(vectorW, omega, vectorU, dimension);
			vector_add(vectorZ, vectorBetaE, MINUS, vectorW, dimension);
			vector_scalar_multiplication(vectorZ, beta, vectorZ, dimension);
			vector_add(vectorVj, vectorSine, PLUS, vectorZ, dimension);

			/* xp_vec <-- xs_method */
			vector_copy(vectorR, vectorSine, dimension);
			vector_copy(vectorBetaE, vectorVj, dimension);
			vector_copy(vectorX_current, vectorX_next, dimension);
		}
		else
		{
			methodError = method_error(barVectorX, vectorX_next, dimension);
		}
	}

	free_doublevector(vectorZ, INT_ONE, dimension);
	free_doublevector(vectorY, INT_ONE, dimension);
	free_doublevector(vectorW, INT_ONE, dimension);
	free_doublevector(vectorV, INT_ONE, dimension);
	free_doublevector(vectorU, INT_ONE, dimension);
	free_doublevector(vectorQj, INT_ONE, dimension);
	free_doublevector(vectorVj, INT_ONE, dimension);
	free_doublevector(vectorBetaE, INT_ONE, dimension);
	free_doublevector(vectorSine, INT_ONE, dimension);
	free_doublevector(vectorCosine, INT_ONE, dimension);
	free_doublevector(vectorR, INT_ONE, dimension);
	free_doublevector(vectorX_next, INT_ONE, dimension);
	free_doublevector(vectorX_current, INT_ONE, dimension);

	return methodError;
}

double restarted_generalize_minimum_residual_method(
	double **matrixA,
	double *vectorB,
	double *barVectorX,
	double *vectorX_initial,
	int restartParameter,
	int maximumRestart,
	double tolerance,
	int dimension)
{
	int matrixDimension = restartParameter, k = 0;

	double beta, kappa = 0.38, methodError = 0.0, normVj = 0.0;

	double **matrixQ, **matrixH;

	double *vectorX_current, *vectorX_next;
	double *vectorR, *vectorSine, *vectorCosine;
	double *vectorBetaE, *vectorVj, *vectorQj;
	double *vectorY, *vectorZ;

	matrixQ = doublematrix(INT_ONE, dimension, INT_ONE, restartParameter + 1);
	matrixH = doublematrix(INT_ONE, restartParameter + 1, INT_ONE, restartParameter);

	vectorX_current = doublevector(INT_ONE, dimension);
	vectorX_next = doublevector(INT_ONE, dimension);
	vectorR = doublevector(INT_ONE, dimension);
	vectorSine = doublevector(INT_ONE, dimension);
	vectorCosine = doublevector(INT_ONE, dimension);
	vectorBetaE = doublevector(INT_ONE, dimension);
	vectorVj = doublevector(INT_ONE, dimension);
	vectorQj = doublevector(INT_ONE, dimension);
	vectorY = doublevector(INT_ONE, dimension);
	vectorZ = doublevector(INT_ONE, dimension);

	vector_copy(vectorX_current, vectorX_initial, dimension);

	// Control variables of while
	bool flag = true;

	while (flag == true)
	{
		matrix_vector_multiplication(vectorR, DOUBLE_ONE, matrixA, vectorX_current, dimension, dimension);
		vector_add(vectorR, vectorB, MINUS, vectorR, dimension);
		beta = vector_norm_2(vectorR, dimension);
		vector_scalar_multiplication(vectorVj, 1 / beta, vectorR, dimension);

		/* Q[:,1] <- v1 */
		vector_to_matrix(vectorVj, matrixQ, 1, dimension);

		/* be1 <- (beta,0,0,...,0) \in R^(k+1)*/
		vectorBetaE[1] = beta;

		for (int j = 1; j <= restartParameter; j++)
		{
			/* Arnoldi's Method */
			matrix_to_vector(matrixQ, vectorQj, j, dimension);
			matrix_vector_multiplication(vectorVj, DOUBLE_ONE, matrixA, vectorQj, dimension, dimension);
			normVj = vector_norm_2(vectorVj, dimension);
			for (int i = 1; i <= j; i++)
			{
				matrix_to_vector(matrixQ, vectorQj, i, dimension);
				matrixH[i][j] = vector_dot(vectorQj, vectorVj, dimension);
				vector_scalar_multiplication(vectorQj, matrixH[i][j], vectorQj, dimension);
				vector_add(vectorVj, vectorVj, MINUS, vectorQj, dimension);
			}
			/* Test for loss of orthogonality and reorthogonalize if necessary */
			if (vector_norm_2(vectorVj, dimension) / normVj <= kappa)
			{
				double p = 0.0;
				for (int i = 1; i <= j; i++)
				{
					matrix_to_vector(matrixQ, vectorQj, i, dimension);
					p = vector_dot(vectorQj, vectorVj, dimension);
					vector_scalar_multiplication(vectorQj, p, vectorQj, dimension);
					vector_add(vectorVj, vectorVj, MINUS, vectorQj, dimension);
					matrixH[i][j] = matrixH[i][j] + p;
				}
			}
			matrixH[j + 1][j] = vector_norm_2(vectorVj, dimension);
			/* Breakdown */
			if (fabs(matrixH[j + 1][j]) <= tolerance)
			{
				matrixDimension = j;
				// printf("successful\n");
				break;
			}

			vector_scalar_multiplication(vectorVj, 1 / matrixH[j + 1][j], vectorVj, dimension);
			vector_to_matrix(vectorVj, matrixQ, j + 1, dimension);

			/* Givens Rotation to find R and be1 to solve Least Square Problem */
			/* compute the product of the previous Givens rotations to the jth column of H */
			double valueHij = 0.0;
			for (int i = 1; i <= j - 1; i++)
			{
				valueHij = matrixH[i][j];
				matrixH[i][j] = vectorCosine[i] * valueHij + vectorSine[i] * matrixH[i + 1][j];
				matrixH[i + 1][j] = -vectorSine[i] * valueHij + vectorCosine[i] * matrixH[i + 1][j];
			}
			/* compute and apply Givens rotation to put 0 in the subdiagonal */
			if (fabs(matrixH[j][j]) > fabs(matrixH[j + 1][j]))
			{
				double t = 0.0, u = 0.0, sign = 0.0;
				t = matrixH[j + 1][j] / matrixH[j][j];
				sign = copysign(1.0, matrixH[j][j]);
				u = sign * sqrt(1 + t * t);
				vectorCosine[j] = 1 / u;
				vectorSine[j] = t * vectorCosine[j];
			}
			else
			{
				double t = 0.0, u = 0.0, sign = 0.0;
				t = matrixH[j][j] / matrixH[j + 1][j];
				sign = copysign(1.0, matrixH[j + 1][j]);
				u = sign * sqrt(1 + t * t);
				vectorSine[j] = 1 / u;
				vectorCosine[j] = t * vectorSine[j];
			}

			/* apply the rotation found (remember that be1[j+1]=0) and do H[j+1,j] = 0 */
			double valueHjj;
			valueHjj = matrixH[j][j];
			matrixH[j][j] = vectorCosine[j] * valueHjj + vectorSine[j] * matrixH[j + 1][j];
			matrixH[j + 1][j] = -vectorSine[j] * valueHjj + vectorCosine[j] * matrixH[j + 1][j];
			vectorBetaE[j + 1] = -vectorSine[j] * vectorBetaE[j];
			vectorBetaE[j] = vectorCosine[j] * vectorBetaE[j];
		}

		/* compute last R column of QR Givens */
		double valueHim = 0.0;
		for (int i = 1; i <= matrixDimension - 1; i++)
		{
			valueHim = matrixH[i][matrixDimension];
			matrixH[i][matrixDimension] = vectorCosine[i] * valueHim + vectorSine[i] * matrixH[i + 1][matrixDimension];
			matrixH[i + 1][matrixDimension] = -vectorSine[i] * valueHim + vectorCosine[i] * matrixH[i + 1][matrixDimension];
		}

		backward_substitution(matrixH, vectorBetaE, vectorY, matrixDimension);

		matrix_vector_multiplication(vectorZ, DOUBLE_ONE, matrixQ, vectorY, dimension, matrixDimension);
		vector_add(vectorX_next, vectorX_current, PLUS, vectorZ, dimension);

		if (k == maximumRestart)
		{
			methodError = method_error(barVectorX, vectorX_next, dimension);
			break;
		}

		vector_copy(vectorX_current, vectorX_next, dimension);
		k = k + 1;
	}

	free_doublevector(vectorZ, INT_ONE, dimension);
	free_doublevector(vectorY, INT_ONE, dimension);
	free_doublevector(vectorQj, INT_ONE, dimension);
	free_doublevector(vectorVj, INT_ONE, dimension);
	free_doublevector(vectorBetaE, INT_ONE, dimension);
	free_doublevector(vectorSine, INT_ONE, dimension);
	free_doublevector(vectorCosine, INT_ONE, dimension);
	free_doublevector(vectorR, INT_ONE, dimension);
	free_doublevector(vectorX_next, INT_ONE, dimension);
	free_doublevector(vectorX_current, INT_ONE, dimension);

	free_doublematrix(matrixH, INT_ONE, restartParameter + 1, INT_ONE, restartParameter);
	free_doublematrix(matrixQ, INT_ONE, dimension, INT_ONE, restartParameter + 1);

	return methodError;
}
