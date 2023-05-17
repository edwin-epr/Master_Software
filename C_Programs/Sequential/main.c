/*
 * barVectorX: vector solución exacta
 */
/* ************************************************************************************************************************ */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
/* ************************************************************************************************************************ */
#include "src/basic_numerical_routines.h"
#include "src/iteratives_solvers.h"
/* ************************************************************************************************************************ */
#define PLUS 1.0
#define MINUS -1.0
#define INT_ONE 1
#define DOUBLE_ONE 1.0
#define EXPERIMENTS_NUMBER 10
#define EIGHT 8
#define BASE_ITERATION_A 25
#define BASE_ITERATION_B 5
/* ************************************************************************************************************************ */
typedef struct
{
	int *vectorIterationNumber;
	double *vectorMethodError;
	double *vectorRunTime;
} numericalExperiments;
/* ************************************************************************************************************************ */
// main function
/* ************************************************************************************************************************ */
int main(int argc, char const *argv[])
{
	int dimension = atoi(argv[1]); // Size of the linear equation system
	double **matrixA, *vectorB, **matrixD, **matrixL, **matrixU;
	double *vectorX_initial, *barVectorX;

	/* allocate vectors and matrices */
	matrixA = doublematrix(INT_ONE, dimension, INT_ONE, dimension);
	system_matrix(matrixA, dimension);

	vectorB = doublevector(INT_ONE, dimension);
	terms_independents_vector(vectorB, dimension);

	barVectorX = doublevector(INT_ONE, dimension); // Exact solution of SEL
	for (int i = 1; i <= dimension; i++)
	{
		barVectorX[i] = 1.0;
	}

	vectorX_initial = doublevector(INT_ONE, dimension);
	for (int i = 1; i <= dimension; i++)
	{
		vectorX_initial[i] = 0.0;
	}

	matrixD = doublematrix(INT_ONE, dimension, INT_ONE, dimension);
	matrixL = doublematrix(INT_ONE, dimension, INT_ONE, dimension);
	matrixU = doublematrix(INT_ONE, dimension, INT_ONE, dimension);

	/************************************************/
	//			 Split A = D + L + U
	/************************************************/

	/* computes D matrix as diagonal of A */
	diagonal_matrix(matrixD, matrixA, dimension);

	/* computes L matrix as a lower triangular part of A */
	lower_triangular_matrix(matrixL, matrixA, dimension);

	/* computes U matrix as a upper triangular part of A */
	upper_triangular_matrix(matrixU, matrixA, dimension);

	/* ******************************************************************* */
	// 							Menu solvers
	/* ******************************************************************* */
	int choice = -1;
	printf("I. Menú de Métodos Iterativos:\n");
	printf("1. Jacobi\n2. Gauss-Seidel\n3. Successive Over-Relaxation\n");
	printf("\nII. Menú de Métodos Proyectivos:\n");
	printf("4. Conjugate Gradient\n5. Bi-Conjugate Gradiente Stabilized\n6. Restarted Generalized Minimal Residual\n");
	printf("\n7. Salir\n");

	while (1)
	{
		printf("\nEnter your choice: ");
		scanf("%d", &choice);

		if (choice == 1)
		{
			/* *********************************************************** */
			// 							Jacobi method
			/* *********************************************************** */
			printf("\nMétodo de Jacobi\n");

			numericalExperiments numericalExperimentsJacobi;
			numericalExperimentsJacobi.vectorIterationNumber = intvector(INT_ONE, EIGHT);
			numericalExperimentsJacobi.vectorMethodError = doublevector(INT_ONE, EIGHT);
			numericalExperimentsJacobi.vectorRunTime = doublevector(INT_ONE, EIGHT);

			double **matrixT, **inverseMatrixD;
			inverseMatrixD = doublematrix(INT_ONE, dimension, INT_ONE, dimension);
			matrixT = doublematrix(INT_ONE, dimension, INT_ONE, dimension);

			/* computes the inverse matrix of D */
			inverse_diagonal_matrix(inverseMatrixD, matrixD, dimension);

			/* computes T = L+U matrix */
			matrix_add(matrixT, matrixL, PLUS, matrixU, dimension);

			for (int i = INT_ONE; i <= EIGHT; i++)
			{
				numericalExperimentsJacobi.vectorIterationNumber[i] = i * BASE_ITERATION_A;
				numericalExperimentsJacobi.vectorMethodError[i] = jacobi_method(inverseMatrixD, matrixT, vectorB, barVectorX, vectorX_initial, numericalExperimentsJacobi.vectorIterationNumber[i], dimension);
			}

			clock_t start_time_count, end_time_count;
			// double cpuTimeJacobi;
			for (int i = INT_ONE; i <= EIGHT; i++)
			{
				double *vectorTimeStep;
				vectorTimeStep = doublevector(INT_ONE, EXPERIMENTS_NUMBER);
				for (int j = 1; j <= EXPERIMENTS_NUMBER; j++)
				{
					start_time_count = clock();
					jacobi_method(inverseMatrixD, matrixT, vectorB, barVectorX, vectorX_initial, numericalExperimentsJacobi.vectorIterationNumber[i], dimension);
					end_time_count = clock();
					vectorTimeStep[j] = ((double)(end_time_count - start_time_count)) / CLOCKS_PER_SEC;
				}

				numericalExperimentsJacobi.vectorRunTime[i] = minimum_value(vectorTimeStep, EXPERIMENTS_NUMBER);

				free_doublevector(vectorTimeStep, INT_ONE, EXPERIMENTS_NUMBER);
			}

			method_information(numericalExperimentsJacobi.vectorMethodError[3], numericalExperimentsJacobi.vectorRunTime[3], numericalExperimentsJacobi.vectorIterationNumber[3]);

			FILE *archivo;

			char *fileName;
			fileName = (char *)malloc(100 * sizeof(char));
			strcpy(fileName, "numerical_experiments/jacobiNumericalResults_");

			char *systemDimension;
			systemDimension = (char *)malloc(6 * sizeof(char));
			sprintf(systemDimension, "%d", dimension);

			strcat(fileName, systemDimension);
			strcat(fileName, ".csv");

			archivo = fopen(fileName, "w");
			if (archivo == NULL)
			{
				printf("\nERROR - No se puede abrir el archivo indicado\n");
			}
			else
			{
				for (int i = INT_ONE; i <= EIGHT; i++)
				{
					fprintf(archivo, "%d,%e,%e\n", numericalExperimentsJacobi.vectorIterationNumber[i], numericalExperimentsJacobi.vectorMethodError[i], numericalExperimentsJacobi.vectorRunTime[i]);
				}

				fclose(archivo);
			}

			free(systemDimension);
			free(fileName);

			free_doublevector(numericalExperimentsJacobi.vectorRunTime, INT_ONE, EIGHT);
			free_doublevector(numericalExperimentsJacobi.vectorMethodError, INT_ONE, EIGHT);
			free_intvector(numericalExperimentsJacobi.vectorIterationNumber, INT_ONE, EIGHT);

			free_doublematrix(matrixT, INT_ONE, dimension, INT_ONE, dimension);
			free_doublematrix(inverseMatrixD, INT_ONE, dimension, INT_ONE, dimension);

			/* *********************************************************** */
		}
		else if (choice == 2)
		{
			/* *********************************************************** */
			// 						Gauss-Seidel method
			/* *********************************************************** */
			printf("\nMétodo de Gauss-Seidel\n");

			numericalExperiments numericalExperimentsGS;
			numericalExperimentsGS.vectorIterationNumber = intvector(INT_ONE, EIGHT);
			numericalExperimentsGS.vectorMethodError = doublevector(INT_ONE, EIGHT);
			numericalExperimentsGS.vectorRunTime = doublevector(INT_ONE, EIGHT);

			double **matrixT, **inverseMatrixT;
			matrixT = doublematrix(INT_ONE, dimension, INT_ONE, dimension);
			inverseMatrixT = doublematrix(INT_ONE, dimension, INT_ONE, dimension);

			/* computes T = D+L matrix */
			matrix_add(matrixT, matrixD, PLUS, matrixL, dimension);

			/* computes the inverse matrix of T */
			inverse(matrixT, inverseMatrixT, dimension);

			for (int i = INT_ONE; i <= EIGHT; i++)
			{
				numericalExperimentsGS.vectorIterationNumber[i] = i * BASE_ITERATION_A;
				numericalExperimentsGS.vectorMethodError[i] = gauss_seidel_method(inverseMatrixT, matrixU, vectorB, barVectorX, vectorX_initial, numericalExperimentsGS.vectorIterationNumber[i], dimension);
			}

			clock_t start_time_count, end_time_count;
			for (int i = INT_ONE; i <= EIGHT; i++)
			{
				double *vectorTimeStep;
				vectorTimeStep = doublevector(INT_ONE, EXPERIMENTS_NUMBER);
				for (int j = 1; j <= EXPERIMENTS_NUMBER; j++)
				{
					start_time_count = clock();
					gauss_seidel_method(inverseMatrixT, matrixU, vectorB, barVectorX, vectorX_initial, numericalExperimentsGS.vectorIterationNumber[i], dimension);
					end_time_count = clock();
					vectorTimeStep[j] = ((double)(end_time_count - start_time_count)) / CLOCKS_PER_SEC;
				}

				numericalExperimentsGS.vectorRunTime[i] = minimum_value(vectorTimeStep, EXPERIMENTS_NUMBER);

				free_doublevector(vectorTimeStep, INT_ONE, EXPERIMENTS_NUMBER);
			}

			method_information(numericalExperimentsGS.vectorMethodError[3], numericalExperimentsGS.vectorRunTime[3], numericalExperimentsGS.vectorIterationNumber[3]);

			FILE *archivo;

			char *fileName;
			fileName = (char *)malloc(100 * sizeof(char));
			strcpy(fileName, "numerical_experiments/gaussSeidelNumericalResults_");

			char *systemDimension;
			systemDimension = (char *)malloc(6 * sizeof(char));
			sprintf(systemDimension, "%d", dimension);

			strcat(fileName, systemDimension);
			strcat(fileName, ".csv");

			archivo = fopen(fileName, "w");
			if (archivo == NULL)
			{
				printf("\nERROR - No se puede abrir el archivo indicado\n");
			}
			else
			{
				for (int i = INT_ONE; i <= EIGHT; i++)
				{
					fprintf(archivo, "%d,%e,%lf\n", numericalExperimentsGS.vectorIterationNumber[i], numericalExperimentsGS.vectorMethodError[i], numericalExperimentsGS.vectorRunTime[i]);
				}

				fclose(archivo);
			}

			free(systemDimension);
			free(fileName);

			free_doublevector(numericalExperimentsGS.vectorRunTime, INT_ONE, EIGHT);
			free_doublevector(numericalExperimentsGS.vectorMethodError, INT_ONE, EIGHT);
			free_intvector(numericalExperimentsGS.vectorIterationNumber, INT_ONE, EIGHT);

			free_doublematrix(inverseMatrixT, INT_ONE, dimension, INT_ONE, dimension);
			free_doublematrix(matrixT, INT_ONE, dimension, INT_ONE, dimension);

			/* *********************************************************** */
		}
		else if (choice == 3)
		{
			/* *********************************************************** */
			// 				Successive Over-Relaxation method
			/* *********************************************************** */
			printf("\nMétodo de Sobre Relajación Sucesiva\n");

			double omega;

			double **matrixOmegaD, **matrixOmegaL, **matrixOmegaU;
			double **matrixT, **inverseMatrixT;
			matrixOmegaD = doublematrix(INT_ONE, dimension, INT_ONE, dimension);
			matrixOmegaL = doublematrix(INT_ONE, dimension, INT_ONE, dimension);
			matrixOmegaU = doublematrix(INT_ONE, dimension, INT_ONE, dimension);
			matrixT = doublematrix(INT_ONE, dimension, INT_ONE, dimension);
			inverseMatrixT = doublematrix(INT_ONE, dimension, INT_ONE, dimension);

			double *vectorE, *vectorF;
			vectorE = doublevector(INT_ONE, dimension);
			vectorF = doublevector(INT_ONE, dimension);

			numericalExperiments numericalExperimentsSOR;
			numericalExperimentsSOR.vectorIterationNumber = intvector(INT_ONE, EIGHT);
			numericalExperimentsSOR.vectorMethodError = doublevector(INT_ONE, EIGHT);
			numericalExperimentsSOR.vectorRunTime = doublevector(INT_ONE, EIGHT);

			printf("\nEnter relaxation parameter: ");
			scanf("%lf", &omega);
			if (omega < 0.0 || omega > 2.0)
			{
				numericalError("Parámetro de relajación está fuera del rango de convergencia...");
			}

			/* computes (1-omega)*D matrix */
			matrix_scalar_multiplication(matrixOmegaD, 1.0 - omega, matrixD, dimension);

			/* computes omega*L matrix */
			matrix_scalar_multiplication(matrixOmegaL, omega, matrixL, dimension);

			/* computes omega*U matrix */
			matrix_scalar_multiplication(matrixOmegaU, omega, matrixU, dimension);

			/* computes D+omega*L matrix */
			matrix_add(matrixT, matrixD, PLUS, matrixOmegaL, dimension);

			/* computes inv(D+omega*L) matrix */
			inverse(matrixT, inverseMatrixT, dimension);

			/* computes omega*b vector */
			vector_scalar_multiplication(vectorE, omega, vectorB, dimension);

			/* computes omega*(D+omega*L)*b vector */
			matrix_vector_multiplication(vectorF, DOUBLE_ONE, inverseMatrixT, vectorE, dimension, dimension);

			for (int i = INT_ONE; i <= EIGHT; i++)
			{
				numericalExperimentsSOR.vectorIterationNumber[i] = i * BASE_ITERATION_A;
				numericalExperimentsSOR.vectorMethodError[i] = sor_method(inverseMatrixT, matrixOmegaD, matrixOmegaU, vectorF, barVectorX, vectorX_initial, numericalExperimentsSOR.vectorIterationNumber[i], dimension);
			}

			clock_t start_time_count, end_time_count;
			for (int i = INT_ONE; i <= EIGHT; i++)
			{
				double *vectorTimeStep;
				vectorTimeStep = doublevector(INT_ONE, EXPERIMENTS_NUMBER);
				for (int j = 1; j <= EXPERIMENTS_NUMBER; j++)
				{
					start_time_count = clock();
					sor_method(inverseMatrixT, matrixOmegaD, matrixOmegaU, vectorF, barVectorX, vectorX_initial, numericalExperimentsSOR.vectorIterationNumber[i], dimension);
					end_time_count = clock();
					vectorTimeStep[j] = ((double)(end_time_count - start_time_count)) / CLOCKS_PER_SEC;
				}

				numericalExperimentsSOR.vectorRunTime[i] = minimum_value(vectorTimeStep, EXPERIMENTS_NUMBER);

				free_doublevector(vectorTimeStep, INT_ONE, EXPERIMENTS_NUMBER);
			}

			method_information(numericalExperimentsSOR.vectorMethodError[3], numericalExperimentsSOR.vectorRunTime[3], numericalExperimentsSOR.vectorIterationNumber[3]);

			FILE *archivo;

			char *fileName;
			fileName = (char *)malloc(100 * sizeof(char));
			strcpy(fileName, "numerical_experiments/sorNumericalResults_");

			char *systemDimension;
			systemDimension = (char *)malloc(6 * sizeof(char));
			sprintf(systemDimension, "%d", dimension);

			char *relaxationParameter;
			relaxationParameter = (char *)malloc(5 * sizeof(char));
			sprintf(relaxationParameter, "%.2lf", omega);

			strcat(fileName, systemDimension);
			strcat(fileName, "_");
			strcat(fileName, relaxationParameter);
			strcat(fileName, ".csv");

			archivo = fopen(fileName, "w");
			if (archivo == NULL)
			{
				printf("\nERROR - No se puede abrir el archivo indicado\n");
			}
			else
			{
				for (int i = INT_ONE; i <= EIGHT; i++)
				{
					fprintf(archivo, "%d,%e,%lf\n", numericalExperimentsSOR.vectorIterationNumber[i], numericalExperimentsSOR.vectorMethodError[i], numericalExperimentsSOR.vectorRunTime[i]);
				}

				fclose(archivo);
			}

			free(relaxationParameter);
			free(systemDimension);
			free(fileName);

			free_doublevector(numericalExperimentsSOR.vectorRunTime, INT_ONE, EIGHT);
			free_doublevector(numericalExperimentsSOR.vectorMethodError, INT_ONE, EIGHT);
			free_intvector(numericalExperimentsSOR.vectorIterationNumber, INT_ONE, EIGHT);

			free_doublevector(vectorF, INT_ONE, dimension);
			free_doublevector(vectorE, INT_ONE, dimension);

			free_doublematrix(inverseMatrixT, INT_ONE, dimension, INT_ONE, dimension);
			free_doublematrix(matrixT, INT_ONE, dimension, INT_ONE, dimension);
			free_doublematrix(matrixOmegaU, INT_ONE, dimension, INT_ONE, dimension);
			free_doublematrix(matrixOmegaL, INT_ONE, dimension, INT_ONE, dimension);
			free_doublematrix(matrixOmegaD, INT_ONE, dimension, INT_ONE, dimension);
			// system("clear");

			/* *********************************************************** */
		}
		else if (choice == 4)
		{
			/* *********************************************************** */
			// 				Conjugate Gradient Method
			/* *********************************************************** */
			printf("\nConjugate Gradient Method\n");

			numericalExperiments numericalExperimentsCG;
			numericalExperimentsCG.vectorIterationNumber = intvector(INT_ONE, EIGHT);
			numericalExperimentsCG.vectorMethodError = doublevector(INT_ONE, EIGHT);
			numericalExperimentsCG.vectorRunTime = doublevector(INT_ONE, EIGHT);

			for (int i = INT_ONE; i <= EIGHT; i++)
			{
				numericalExperimentsCG.vectorIterationNumber[i] = i * BASE_ITERATION_B;
				numericalExperimentsCG.vectorMethodError[i] = conjugate_gradient_method(matrixA, vectorB, barVectorX, vectorX_initial, numericalExperimentsCG.vectorIterationNumber[i], dimension);
			}

			clock_t start_time_count, end_time_count;
			for (int i = INT_ONE; i <= EIGHT; i++)
			{
				double *vectorTimeStep;
				vectorTimeStep = doublevector(INT_ONE, EXPERIMENTS_NUMBER);
				for (int j = 1; j <= EXPERIMENTS_NUMBER; j++)
				{
					start_time_count = clock();
					conjugate_gradient_method(matrixA, vectorB, barVectorX, vectorX_initial, numericalExperimentsCG.vectorIterationNumber[i], dimension);
					end_time_count = clock();
					vectorTimeStep[j] = ((double)(end_time_count - start_time_count)) / CLOCKS_PER_SEC;
				}

				numericalExperimentsCG.vectorRunTime[i] = minimum_value(vectorTimeStep, EXPERIMENTS_NUMBER);

				free_doublevector(vectorTimeStep, INT_ONE, EXPERIMENTS_NUMBER);
			}

			method_information(numericalExperimentsCG.vectorMethodError[3], numericalExperimentsCG.vectorRunTime[3], numericalExperimentsCG.vectorIterationNumber[3]);

			FILE *archivo;

			char *fileName;
			fileName = (char *)malloc(100 * sizeof(char));
			strcpy(fileName, "numerical_experiments/conjugateGradientNumericalResults_");

			char *systemDimension;
			systemDimension = (char *)malloc(6 * sizeof(char));
			sprintf(systemDimension, "%d", dimension);

			strcat(fileName, systemDimension);
			strcat(fileName, ".csv");

			archivo = fopen(fileName, "w");
			if (archivo == NULL)
			{
				printf("\nERROR - No se puede abrir el archivo indicado\n");
			}
			else
			{
				for (int i = INT_ONE; i <= EIGHT; i++)
				{
					fprintf(archivo, "%d,%e,%lf\n", numericalExperimentsCG.vectorIterationNumber[i], numericalExperimentsCG.vectorMethodError[i], numericalExperimentsCG.vectorRunTime[i]);
				}

				fclose(archivo);
			}

			free(systemDimension);
			free(fileName);

			free_doublevector(numericalExperimentsCG.vectorRunTime, INT_ONE, EIGHT);
			free_doublevector(numericalExperimentsCG.vectorMethodError, INT_ONE, EIGHT);
			free_intvector(numericalExperimentsCG.vectorIterationNumber, INT_ONE, EIGHT);

			/* *********************************************************** */
		}
		else if (choice == 5)
		{
			/* *********************************************************** */
			// * 		Biconjugate Gradient Stabilized Method
			/* *********************************************************** */
			printf("\nBiConjugate Gradient Stabilized Method\n");

			numericalExperiments numericalExperimentsBiCGSTAB;
			numericalExperimentsBiCGSTAB.vectorIterationNumber = intvector(INT_ONE, EIGHT);
			numericalExperimentsBiCGSTAB.vectorMethodError = doublevector(INT_ONE, EIGHT);
			numericalExperimentsBiCGSTAB.vectorRunTime = doublevector(INT_ONE, EIGHT);

			for (int i = INT_ONE; i <= EIGHT; i++)
			{
				numericalExperimentsBiCGSTAB.vectorIterationNumber[i] = i * BASE_ITERATION_B;
				numericalExperimentsBiCGSTAB.vectorMethodError[i] = biconjugate_gradient_stabilized_method(matrixA, vectorB, barVectorX, vectorX_initial, numericalExperimentsBiCGSTAB.vectorIterationNumber[i], dimension);
			}

			clock_t start_time_count, end_time_count;
			for (int i = INT_ONE; i <= EIGHT; i++)
			{
				double *vectorTimeStep;
				vectorTimeStep = doublevector(INT_ONE, EXPERIMENTS_NUMBER);
				for (int j = 1; j <= EXPERIMENTS_NUMBER; j++)
				{
					start_time_count = clock();
					biconjugate_gradient_stabilized_method(matrixA, vectorB, barVectorX, vectorX_initial, numericalExperimentsBiCGSTAB.vectorIterationNumber[i], dimension);
					end_time_count = clock();
					vectorTimeStep[j] = ((double)(end_time_count - start_time_count)) / CLOCKS_PER_SEC;
				}

				numericalExperimentsBiCGSTAB.vectorRunTime[i] = minimum_value(vectorTimeStep, EXPERIMENTS_NUMBER);

				free_doublevector(vectorTimeStep, INT_ONE, EXPERIMENTS_NUMBER);
			}

			method_information(numericalExperimentsBiCGSTAB.vectorMethodError[3], numericalExperimentsBiCGSTAB.vectorRunTime[3], numericalExperimentsBiCGSTAB.vectorIterationNumber[3]);

			FILE *archivo;

			char *fileName;
			fileName = (char *)malloc(100 * sizeof(char));
			strcpy(fileName, "numerical_experiments/biconjugateGradientStabilizedNumericalResults_");

			char *systemDimension;
			systemDimension = (char *)malloc(6 * sizeof(char));
			sprintf(systemDimension, "%d", dimension);

			strcat(fileName, systemDimension);
			strcat(fileName, ".csv");

			archivo = fopen(fileName, "w");
			if (archivo == NULL)
			{
				printf("\nERROR - No se puede abrir el archivo indicado\n");
			}
			else
			{
				for (int i = INT_ONE; i <= EIGHT; i++)
				{
					fprintf(archivo, "%d,%e,%lf\n", numericalExperimentsBiCGSTAB.vectorIterationNumber[i], numericalExperimentsBiCGSTAB.vectorMethodError[i], numericalExperimentsBiCGSTAB.vectorRunTime[i]);
				}

				fclose(archivo);
			}

			free(systemDimension);
			free(fileName);

			free_doublevector(numericalExperimentsBiCGSTAB.vectorRunTime, INT_ONE, EIGHT);
			free_doublevector(numericalExperimentsBiCGSTAB.vectorMethodError, INT_ONE, EIGHT);
			free_intvector(numericalExperimentsBiCGSTAB.vectorIterationNumber, INT_ONE, EIGHT);

			/* *********************************************************** */
		}
		else if (choice == 6)
		{
			/* *********************************************************** */
			// * 	Restarted Generalized Minimal Residual Method
			/* *********************************************************** */
			printf("\nRestarted Generalized Minimal Residual Method\n");

			const double tolerance = 1e-10;
			int restartParameter;

			printf("\nEnter restart parameter: ");
			scanf("%d", &restartParameter);

			numericalExperiments numericalExperimentsGMRES;
			numericalExperimentsGMRES.vectorIterationNumber = intvector(INT_ONE, EIGHT);
			numericalExperimentsGMRES.vectorMethodError = doublevector(INT_ONE, EIGHT);
			numericalExperimentsGMRES.vectorRunTime = doublevector(INT_ONE, EIGHT);

			for (int i = INT_ONE; i <= EIGHT; i++)
			{
				numericalExperimentsGMRES.vectorIterationNumber[i] = i * BASE_ITERATION_B;
				numericalExperimentsGMRES.vectorMethodError[i] = restarted_generalize_minimum_residual_method(matrixA, vectorB, barVectorX, vectorX_initial, restartParameter, numericalExperimentsGMRES.vectorIterationNumber[i], tolerance, dimension);
			}

			clock_t start_time_count, end_time_count;
			for (int i = INT_ONE; i <= EIGHT; i++)
			{
				double *vectorTimeStep;
				vectorTimeStep = doublevector(INT_ONE, EXPERIMENTS_NUMBER);
				for (int j = 1; j <= EXPERIMENTS_NUMBER; j++)
				{
					start_time_count = clock();
					restarted_generalize_minimum_residual_method(matrixA, vectorB, barVectorX, vectorX_initial, restartParameter, numericalExperimentsGMRES.vectorIterationNumber[i], tolerance, dimension);
					end_time_count = clock();
					vectorTimeStep[j] = ((double)(end_time_count - start_time_count)) / CLOCKS_PER_SEC;
				}

				numericalExperimentsGMRES.vectorRunTime[i] = minimum_value(vectorTimeStep, EXPERIMENTS_NUMBER);

				free_doublevector(vectorTimeStep, INT_ONE, EXPERIMENTS_NUMBER);
			}

			method_information(numericalExperimentsGMRES.vectorMethodError[3], numericalExperimentsGMRES.vectorRunTime[3], numericalExperimentsGMRES.vectorIterationNumber[3]);

			FILE *archivo;

			char *fileName;
			fileName = (char *)malloc(100 * sizeof(char));
			strcpy(fileName, "numerical_experiments/restartedGMRESNumericalResults_");

			char *systemDimension;
			systemDimension = (char *)malloc(6 * sizeof(char));
			sprintf(systemDimension, "%d", dimension);

			char *restartParameterString;
			restartParameterString = (char *)malloc(3 * sizeof(char));
			sprintf(restartParameterString, "%d", restartParameter);

			strcat(fileName, systemDimension);
			strcat(fileName, "_");
			strcat(fileName, restartParameterString);
			strcat(fileName, ".csv");

			archivo = fopen(fileName, "w");
			if (archivo == NULL)
			{
				printf("\nERROR - No se puede abrir el archivo indicado\n");
			}
			else
			{
				for (int i = INT_ONE; i <= EIGHT; i++)
				{
					fprintf(archivo, "%d,%e,%lf\n", numericalExperimentsGMRES.vectorIterationNumber[i], numericalExperimentsGMRES.vectorMethodError[i], numericalExperimentsGMRES.vectorRunTime[i]);
				}

				fclose(archivo);
			}

			free(restartParameterString);
			free(systemDimension);

			free_doublevector(numericalExperimentsGMRES.vectorRunTime, INT_ONE, EIGHT);
			free_doublevector(numericalExperimentsGMRES.vectorMethodError, INT_ONE, EIGHT);
			free_intvector(numericalExperimentsGMRES.vectorIterationNumber, INT_ONE, EIGHT);

			/* *********************************************************** */
		}

		else
		{
			// system("clear");
			printf("\nSee you later...");
			break;
		}
	}

	free_doublematrix(matrixA, INT_ONE, dimension, INT_ONE, dimension);
	free_doublematrix(matrixD, INT_ONE, dimension, INT_ONE, dimension);
	free_doublematrix(matrixL, INT_ONE, dimension, INT_ONE, dimension);
	free_doublematrix(matrixU, INT_ONE, dimension, INT_ONE, dimension);

	free_doublevector(barVectorX, INT_ONE, dimension);
	free_doublevector(vectorX_initial, INT_ONE, dimension);
	free_doublevector(vectorB, INT_ONE, dimension);
	return 0;
}