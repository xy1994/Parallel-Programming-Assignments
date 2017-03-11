// Distributed two-dimensional Discrete FFT transform
// YI XIE
// ECE8893 Project 1


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>

#include "Complex.h"
#include "InputImage.h"


using namespace std;

void Transform1D(Complex*,int,Complex*);
void InvTransform1D(Complex*,int,Complex*);
void Transpose(Complex*,int,int);

void Transform2D(const char* inputFN) 
{ // Do the 2D transform here.
  // 1) Use the InputImage object to read in the Tower.txt file and
  //    find the width/height of the input image.
  // 2) Use MPI to find how many CPUs in total, and which one
  //    this process is
  // 3) Allocate an array of Complex object of sufficient size to
  //    hold the 2d DFT results (size is width * height)
  // 4) Obtain a pointer to the Complex 1d array of input data
  // 5) Do the individual 1D transforms on the rows assigned to your CPU
  // 6) Send the resultant transformed values to the appropriate
  //    other processors for the next phase.
  // 6a) To send and receive columns, you might need a separate
  //     Complex array of the correct size.
  // 7) Receive messages from other processes to collect your columns
  // 8) When all columns received, do the 1D transforms on the columns
  // 9) Send final answers to CPU 0 (unless you are CPU 0)
  //   9a) If you are CPU 0, collect all values from other processors
  //       and print out with SaveImageData().
  InputImage image(inputFN);  // Create the helper object for reading the image
  // Step (1) in the comments is the line above.
  // Your code here, steps 2-9
  int width = image.GetWidth();
  int height = image.GetHeight();
  Complex* data = image.GetImageData();
  
  int numtasks, rank;
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  printf("Number of CPU is %d, the rank of this CPU is %d\n",numtasks,rank);

  int RowsPerCPU = height/numtasks;
  int StartingRow = RowsPerCPU*rank;
  int BulkSize = RowsPerCPU*width*sizeof(Complex);
  
  Complex* outputdata = new Complex[width*height];
  MPI_Status* thestatus = new MPI_Status[numtasks];
  MPI_Request* therequest = new MPI_Request[numtasks];

  for(int i = 0;i<RowsPerCPU;i++){
    Transform1D(data+(StartingRow*width)+(i*width),width,outputdata+(StartingRow*width)+(i*width));
    cout<<"the "<<i<<"th CPU is working on 1d fft..."<<endl;
  }
  
  if(rank != 0){
    MPI_Isend(outputdata+StartingRow*width,BulkSize,MPI_CHAR,0,0,MPI_COMM_WORLD,&therequest[rank]);
    cout<<"the "<<rank<<"CPU is sending 1d fft result to 0..,"<<endl;
  }

  if(rank == 0){
    for(int cpu = 1;cpu<numtasks;cpu++){
      MPI_Recv(data+RowsPerCPU*cpu*width,BulkSize,MPI_CHAR,cpu,0,MPI_COMM_WORLD,&thestatus[cpu]);
      cout<<"the CPU 0 is receiving 1d fft result from the "<<cpu<<"th CPU..."<<endl;
    }

    memcpy(data,outputdata,BulkSize);
    image.SaveImageData("MyAfter1D.txt",data,width,height);
    Transpose(data,width,height);
    } 

  int temp = width;
  width = height;
  height = temp;

  RowsPerCPU = height/numtasks;
 
  StartingRow = RowsPerCPU*rank;
  thestatus = new MPI_Status[numtasks];
  therequest = new MPI_Request[numtasks];
  BulkSize = RowsPerCPU*width*sizeof(Complex);

  if(rank == 0){
    for(int cpu = 0;cpu<numtasks;cpu++){
      MPI_Isend(data+RowsPerCPU*cpu*width,BulkSize,MPI_CHAR,cpu,cpu*RowsPerCPU,MPI_COMM_WORLD,&therequest[cpu]);
      cout<<"CPU 0  is sending columns to the "<<cpu<<"th cpu"<<endl;
   }
  }

  Complex* temph = new Complex[RowsPerCPU*width];
  Complex* tempH = new Complex[RowsPerCPU*width];

      MPI_Recv(temph,BulkSize,MPI_CHAR,0,rank*RowsPerCPU,MPI_COMM_WORLD,&thestatus[rank]);
      cout<<"The "<<rank<<"th cpu is receiving the column"<<endl;

    for(int i = 0;i<RowsPerCPU;i++){
      Transform1D(temph+i*width,width,tempH+i*width);
    }

      MPI_Isend(tempH,BulkSize,MPI_CHAR,0,rank*RowsPerCPU,MPI_COMM_WORLD,&therequest[rank]);
      cout<<"The "<<rank<<"th cpu is sending 2d fft result back to cpu 0"<<endl;


    if(rank == 0){
      for(int cpu = 0;cpu<numtasks;cpu++){
    MPI_Recv(data+RowsPerCPU*cpu*width,BulkSize,MPI_CHAR,cpu,cpu*RowsPerCPU,MPI_COMM_WORLD,&thestatus[cpu]);
    cout<<"The cpu 0 is receiving 2d fft result from the "<<cpu<<"th cpu"<<endl; 
      }

    Transpose(data,width,height);
        temp = width;
        width = height;
    height = temp;
    image.SaveImageData("MyAfter2D.txt",data,width,height);
    }

       
    RowsPerCPU = height/numtasks;
    StartingRow = rank*RowsPerCPU;
    outputdata = new Complex[width*height];
    BulkSize = RowsPerCPU*width*sizeof(Complex);

    thestatus = new MPI_Status[numtasks];
    therequest = new MPI_Request[numtasks];
    
    
    if(rank == 0){
      for(int cpu = 1;cpu<numtasks;cpu++){
    MPI_Isend(data+RowsPerCPU*cpu*width,BulkSize,MPI_CHAR,cpu,0,MPI_COMM_WORLD,&therequest[cpu]);
    cout<<"The cpu 0 is sending transformed image to the "<<cpu<<"th cpu"<<endl;
      }
    }

    if(rank!=0){
    MPI_Recv(data+StartingRow*width,BulkSize,MPI_CHAR,0,0,MPI_COMM_WORLD,&thestatus[rank]);
    cout<<"The "<<rank<<"th cpu is receiving the transformed image"<<endl; 
    }
    

    for(int i = 0;i<RowsPerCPU;i++){
      InvTransform1D(data+(StartingRow*width)+(i*width),width,outputdata+(StartingRow*width)+(i*width));
}


    if(rank!=0){
      MPI_Isend(outputdata+StartingRow*width,BulkSize,MPI_CHAR,0,0,MPI_COMM_WORLD,&therequest[rank]);
      cout<<"The "<<rank<<"th cpu is sending the 1d ifft result to cpu 0"<<endl;
    }


if(rank == 0){
  for(int cpu = 1;cpu<numtasks;cpu++){
    MPI_Recv(data+RowsPerCPU*cpu*width,BulkSize,MPI_CHAR,cpu,0,MPI_COMM_WORLD,&thestatus[cpu]);
    cout<<"The cpu 0 is receiving 1d ifft result from the "<<cpu<<"th cpu"<<endl;
  }
   memcpy(data,outputdata,BulkSize);
  Transpose(data,width,height);
  
 }

 temp = width;
 width = height;
 height = temp;

 RowsPerCPU = height/numtasks;
 StartingRow = RowsPerCPU*rank;
 BulkSize = RowsPerCPU*width*sizeof(Complex);
    


  if(rank == 0){
   for(int cpu = 0;cpu<numtasks;cpu++){
     MPI_Isend(data+RowsPerCPU*cpu*width,BulkSize,MPI_CHAR,cpu,cpu*RowsPerCPU,MPI_COMM_WORLD,&therequest[cpu]);
     cout<<"The cpu 0 is sending column to the "<<cpu<<"th cpu"<<endl;
   }
 }

 temph = new Complex[RowsPerCPU*width];
 tempH = new Complex[RowsPerCPU*width];

 MPI_Recv(temph,BulkSize,MPI_CHAR,0,rank*RowsPerCPU,MPI_COMM_WORLD,&thestatus[rank]);
 cout<<"The "<<rank<<"th cpu is receiving the column"<<endl;

 for(int i = 0;i<RowsPerCPU;i++){
   InvTransform1D(temph+i*width,width,tempH+i*width);
 }

 MPI_Isend(tempH,BulkSize,MPI_CHAR,0,rank*RowsPerCPU,MPI_COMM_WORLD,&therequest[rank]);
 cout<<"The "<<rank<<"th cpu is sending the 2d ifft result back to cpu 0"<<endl;

 if(rank == 0){
   for(int cpu = 0;cpu<numtasks;cpu++){
     MPI_Recv(data+RowsPerCPU*cpu*width,BulkSize,MPI_CHAR,cpu,cpu*RowsPerCPU,MPI_COMM_WORLD,&thestatus[cpu]);
     cout<<"the cpu 0 is receiving the 2d ifft result from the "<<cpu<<"cpu"<<endl;
   }
   Transpose(data,width,height);
   temp = width;
   width = height;
   height = temp;
   image.SaveImageData("MyAfterInverse.txt",data,width,height);
   } 
}

void Transform1D(Complex* h, int w, Complex* H)
{
  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
  // data, w is the width (N), and H is the output array.
  for(int k = 0;k<w;k++){
    for(int n = 0;n<w;n++){
      Complex temp (cos(2*M_PI*n*k/w),-sin(2*M_PI*n*k/w));
      H[k] = H[k] + temp*h[n];
    }
  }
}

void InvTransform1D(Complex* H,int w,Complex* h){
  for(int n = 0;n<w;n++){
    for(int k = 0;k<w;k++){
      Complex temp (cos(2*M_PI*n*k/w),sin(2*M_PI*n*k/w));
      h[n] = h[n] + Complex(1.0/w)*temp*H[k];
    }
  }
}

void Transpose(Complex* h, int width, int height){
  for(int i = 0;i<width;i++){
    for(int j = 0;j<height;j++){
      if(j<=i){
    Complex temp = h[width*j+i];
    h[width*j+i] = h[width*i+j];
    h[width*i+j] = temp;
      }
    }
  }
  }



int main(int argc, char** argv)
{
   string fn("Tower.txt"); // default file name
   if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  // MPI initialization here
 
     int rc =  MPI_Init(&argc,&argv);
     if(rc!=MPI_SUCCESS){
       printf("Error occurs when starting the MPI program!\n");
       MPI_Abort(MPI_COMM_WORLD,rc);
    }  
     Transform2D(fn.c_str()); // Perform the transform.
  // Finalize MPI here
     MPI_Finalize(); 
      return 0;
}  
  

  
