/*
 * CX 4220 / CSE 6220 Introduction to High Performance Computing
 *              Programming Assignment 1
 * 
 *  MPI polynomial evaluation algorithm function implementations go here
 * 
 */

#include "mpi_evaluator.h"
#include "const.h"
#include "math.h"
#include "stdio.h"
#include <stdlib.h>

void scatter(const int n, double* scatter_values, int &n_local, double* &local_values, int source_rank, const MPI_Comm comm){
    //Implementation
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    int p1, p2, n1, n2;

    MPI_Status stat;
    MPI_Request req;

	//Sending the length of total array from source_rank
    int total = 0;
    if (rank == source_rank){
        total = n;
        for (int i = 0; i<source_rank; i++){
            MPI_Send(&total, 1, MPI_INT, i, 111, comm);
        }
        for (int i = source_rank + 1; i<p; i++){
            MPI_Send(&total, 1, MPI_INT, i, 111, comm);
        }
    }

	//Receiving the length of total array from source_rank
    if (rank != source_rank){
        MPI_Recv(&total, 1, MPI_INT, source_rank, 111, comm, &stat);
    }

    n1 = ceil((total*1.0) / p); //Bigger count
    n2 = floor((total*1.0) / p); //Smaller count
    p2 = p * n1 - total; //Number of processors with local count n1
    p1 = p - p2; //Number of processors with local count n2


	//Sending the local values from source_rank
    if (rank == source_rank){
        for (int i = 1; i < source_rank; i++){
            if (i >= p1){
                MPI_Send(&scatter_values[p1 * n1 + (i - p1)*n2], n2, MPI_DOUBLE, i, 222, comm);
            }
            else{
                MPI_Send(&scatter_values[i*n1], n1, MPI_DOUBLE, i, 222, comm);
            }
        }
        for (int i = source_rank + 1; i < p; i++){
            if (i >= p1){
                MPI_Send(&scatter_values[p1 * n1 + (i - p1)*n2], n2, MPI_DOUBLE, i, 222, comm);
            }
            else{
                MPI_Send(&scatter_values[i*n1], n1, MPI_DOUBLE, i, 222, comm);
            }
        }

    }

    if (rank >= p1){
        n_local = n2;
    }
    else{
        n_local = n1;
    }

	//Allocate the space for local values
    local_values = (double*)malloc(sizeof(double)*n_local);

	//Generate local values in source_rank
    if (rank == source_rank){
        for (int i = 0; i < n_local; i++){
            local_values[i] = scatter_values[i];
        }
    }

	//Receiving the local values from source_rank
    if (rank != source_rank){
        MPI_Recv(&local_values[0], n_local, MPI_DOUBLE, source_rank, 222, comm, &stat);
    }
}

double broadcast(double value, int source_rank, const MPI_Comm comm){
    //Implementation
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    int n = ceil(log2(p));
    MPI_Status stat;
    int dis = 1;
    int partner;

    for (int i = 0; i<n; i++){
        int temp = ((rank - source_rank) >= 0) ? (rank - source_rank) : (rank - source_rank + p);
        if (temp <= (2 * dis - 1)){
            if (temp<dis){
                partner = (rank + dis) % p;
				//Broadcasting the value
                if (((rank + dis - source_rank) < p) && ((rank + dis - source_rank) >= 0)){
                    MPI_Send(&value, 1, MPI_DOUBLE, partner, 333, comm);
                }
            }
            else{
                partner = (rank - dis + p) % p;
				int temp = ((rank - dis) < 0) ? (rank - dis + p) : (rank - dis);
                if ((temp<p) && (temp>=0)){
					//Receiving the value
                    MPI_Recv(&value, 1, MPI_DOUBLE, partner, 333, comm, &stat);
                }
            }
        }
        dis = dis * 2;
    }

    return value;
}

void parallel_prefix(const int n, const double* values, double* prefix_results, const int OP, const MPI_Comm comm){
    //Implementation
    double* local_results = (double*)malloc(sizeof(double)*n);
    *(local_results) = *(values);

    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

	//Performing the prefix product
    if (OP == PREFIX_OP_PRODUCT){
        for (int i = 1; i<n; i++){
            *(local_results + i) = *(local_results + i - 1) * (*(values + i));
        }

        int iter = ceil(log2(p));
        int dis = 1;
        double prefix_sum = 1;
        double total_sum = *(local_results + n - 1);
        int flag = 1;
        int partner;
        MPI_Status stat;
        for (int i = 0; i<iter; i++){
            if ((rank & flag) == 0){
                double temp = 1;
                partner = rank + dis;
                if (partner<p){
					//The previous processor
                    MPI_Send(&total_sum, 1, MPI_DOUBLE, partner, 444, comm);
                    MPI_Recv(&temp, 1, MPI_DOUBLE, partner, 555, comm, &stat);
                    total_sum = total_sum * temp;
                }
            }
            else{
                double temp = 1;
                partner = rank - dis;
				//The latter processor
                MPI_Recv(&temp, 1, MPI_DOUBLE, partner, 444, comm, &stat);
                MPI_Send(&total_sum, 1, MPI_DOUBLE, partner, 555, comm);
                total_sum = total_sum * temp;
                prefix_sum = prefix_sum * temp;
            }
            flag = flag << 1;
            dis = dis * 2;
        }

        for (int i = 0; i < n; i++){
			//Update the previous prefix results
            *(local_results + i) = prefix_sum *(*(local_results + i));

            *(prefix_results + i) = *(local_results + i);
        }
    }

	//Performing the prefix sum
    if (OP == PREFIX_OP_SUM){
        for (int i = 1; i<n; i++){
            *(local_results + i) = *(local_results + i - 1) + (*(values + i));
        }
        int iter = ceil(log2(p));
        int dis = 1;
        double prefix_sum = 0;
        double total_sum = *(local_results + n - 1);
        int flag = 1;
        int partner;
        MPI_Status stat;
        for (int i = 0; i<iter; i++){
            if ((rank & flag) == 0){
                double temp = 1;
                partner = rank + dis;
                if (partner<p){
					//The previous processor
                    MPI_Send(&total_sum, 1, MPI_DOUBLE, partner, 666, comm);
                    MPI_Recv(&temp, 1, MPI_DOUBLE, partner, 777, comm, &stat);
                    total_sum = total_sum + temp;
                }
            }
            else{
                double temp = 1;
                partner = rank - dis;
				//The latter processor
                MPI_Recv(&temp, 1, MPI_DOUBLE, partner, 666, comm, &stat);
                MPI_Send(&total_sum, 1, MPI_DOUBLE, partner, 777, comm);
                total_sum = total_sum + temp;
                prefix_sum = prefix_sum + temp;
            }
            flag = flag << 1;
            dis = dis * 2;
        }
        for (int i = 0; i < n; i++){
			//Update the previous prefix results
            *(local_results + i) = prefix_sum + *(local_results + i); 
            *(prefix_results + i) = *(local_results + i);
        }
    }
}

double mpi_poly_evaluator(const double x, const int n, const double* constants, const MPI_Comm comm){
    //Implementation
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);
    MPI_Status stat;

	//Allocate the space
    double* values = (double*)malloc(sizeof(double)*n);
    double* prefix_results = (double*)malloc(sizeof(double)*n);

	//Initialize the prefix result
    if (rank == 0){
        *values = 1;
        for (int i = 1; i < n; i++){
            *(values + i) = x;
        }
    }
    else{
        for (int i = 0; i < n; i++){
            *(values + i) = x;
        }
    }

    for (int i = 0; i<n; i++){
        *(prefix_results + i) = 1;
    }

    int OP = 2;

	//Prefix product
    parallel_prefix(n, values, prefix_results, OP, comm);

    for (int i = 0; i<n; i++){
        *(values + i) = *(prefix_results + i) * (*(constants + i));
        *(prefix_results + i) = 0;
    }

    OP = 1;

	//Prefix sum
    parallel_prefix(n, values, prefix_results, OP, comm);

    double result = 0;

	//Processor 0 receiving the final result
    if (rank == (p - 1)){
        double sum = *(prefix_results + n - 1);
        MPI_Send(&sum, 1, MPI_DOUBLE, 0, 888, comm);
    }
    else if (rank == 0){
        MPI_Recv(&result, 1, MPI_DOUBLE, p-1 , 888, comm, &stat);
    }
    return result;
}