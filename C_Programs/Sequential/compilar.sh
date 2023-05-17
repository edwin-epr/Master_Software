# chmod +x file_name.sh
gcc -Wall -c src/basic_numerical_routines.c
gcc -Wall -c src/iteratives_solvers.c
gcc -Wall -c main.c
gcc main.o basic_numerical_routines.o iteratives_solvers.o -lm -o main