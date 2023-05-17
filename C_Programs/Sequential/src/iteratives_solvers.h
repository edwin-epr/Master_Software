double jacobi_method(
	double **matrixInverseD,
	double **matrixT,
	double *vectorB,
	double *vectorX_exactSolution,
	double *vectorX_initial,
	int maximumIteration,
	int dimension);

double gauss_seidel_method(
	double **inverseMatrixT,
	double **matrixU,
	double *vectorB,
	double *vectorX_exactSolution,
	double *vectorX_initial,
	int maximumIteration,
	int dimension);

double sor_method(
	double **inverseMatrixT,
	double **matrixOmegaD,
	double **matrixOmegaU,
	double *vectorF,
	double *vectorX_exactSolution,
	double *vectorX_initial,
	int maximumIteration,
	int dimension);

double conjugate_gradient_method(
	double **matrixA,
	double *vectorB,
	double *vectorX_exactSolution,
	double *vectorX_initial,
	int maximumIteration,
	int dimension);

double biconjugate_gradient_stabilized_method(
	double **matrixA,
	double *vectorB,
	double *barVectorX,
	double *vectorX_initial,
	int maximumIteration,
	int dimension);

double restarted_generalize_minimum_residual_method(
	double **matrixA,
	double *vectorB,
	double *barVectorX,
	double *vectorX_initial,
	int restartParameter,
	int maximumRestart,
	double tolerance,
	int dimension);
