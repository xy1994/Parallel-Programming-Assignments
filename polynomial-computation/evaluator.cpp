/*
 * CX 4220 / CSE 6220 Introduction to High Performance Computing
 *              Programming Assignment 1
 * 
 *  Serial polynomial evaluation algorithm function implementations goes here
 * 
 */

#include "mpi_evaluator.h"
#include "const.h"
#include "math.h"
#include "stdio.h"
#include <stdlib.h>

double poly_evaluator(const double x, const int n, const double* constants){
    //Implementation
	double* results = (double*)malloc(sizeof(double)*n);
	double* values = (double*)malloc(sizeof(double)*n);
	double sum = 0;

	//Compute the local sum sequentially
	for (int i = 0; i < n; i++){
		*(results + i) = pow(x, i);
	}

	for (int i = 0; i < n; i++){
		*(values + i) = *(results + i) * (*(constants + i));
	}

	for(int j = 0;j < n; j++){
		sum = sum + *(values + j);
	}

	return sum;
}