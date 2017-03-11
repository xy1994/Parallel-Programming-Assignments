// Distributed two-dimensional Discrete FFT transform
// YOUR NAME HERE
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

void Transform1D(Complex* h, int w, Complex* H)
{
  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
  // data, w is the width (N), and H is the output array.
  for ( int ix = 0; ix < w; ++ix )
    for ( int k = 0; k < w; ++k )
      H[ix] = H[ix] + Complex(cos(2*M_PI * ix*k / w), -sin(2*M_PI * ix*k / w)) * h[k];
}

void Transpose(Complex* h, int width, int height)
{
  for( int row = 0; row < height; ++row )
    for( int col = 0; col < width; ++col )
      if( col > row)
      {
        Complex temp = h[row * width + col];
        h[row * width + col] = h[col * width + row];
        h[col * width + row] = temp;
      }
}

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

  int width = image.GetWidth();
  int height = image.GetHeight();
  
  int nCPUs, rank;

  MPI_Comm_size(MPI_COMM_WORLD, &nCPUs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  Complex* H = new Complex[width*height]();
  Complex* T = new Complex[width*height]();
  Complex* R = new Complex[width*height]();
  Complex* h = image.GetImageData();

  int rowsPerCPU = height/nCPUs;

  MPI_Status* pStat = new MPI_Status[nCPUs];
  MPI_Request* pReq = new MPI_Request[nCPUs];

  for ( int row = 0; row < rowsPerCPU; ++row )
  {
    Complex* currenth = h + rank * rowsPerCPU * width + row * width;
    Complex* currentH = H + rank * rowsPerCPU * width + row * width;
    Transform1D(currenth, width, currentH);
  }

    int startRow = rank * rowsPerCPU;

    MPI_Isend(H + startRow * width, rowsPerCPU * width * sizeof(Complex), MPI_CHAR, 0, startRow, MPI_COMM_WORLD, &pReq[rank] );
    cout << "CPU " << rank << " queued send" << endl;

  if( rank == 0 )
  {
    for( int cpu = 0; cpu < nCPUs; ++cpu)
    {
      int startRow = cpu * rowsPerCPU;
      MPI_Recv(T + startRow * width, rowsPerCPU * width * sizeof(Complex), MPI_CHAR, cpu, startRow, MPI_COMM_WORLD, &pStat[cpu]);
      cout << "CPU " << cpu << " queued recv" << endl; 
    }
    cout << "CPU " << rank << " queued all recv" << endl;
  }
  


  if( rank == 0 )
  {
    string fn1("myafter1d.txt");
    image.SaveImageData(fn1.c_str(), T, width, height); 

  }
  
  
  int temp = height;
  height = width;
  width = temp;

  rowsPerCPU = height/nCPUs;
  

  if(rank == 0)
  {
    Transpose(T, width, height);
    for( int cpu = 0; cpu < nCPUs; ++cpu)
    {
      int startRow = cpu * rowsPerCPU;
      MPI_Isend(T + startRow * width, rowsPerCPU * width * sizeof(Complex), MPI_CHAR, cpu, 0, MPI_COMM_WORLD,&pReq[rank]);
      cout << "CPU " << cpu << " queued2 send" << endl; 
    }
  }

    Complex* hh = new Complex[rowsPerCPU * height];
    Complex* HH = new Complex[rowsPerCPU * height];
    MPI_Recv(hh, rowsPerCPU * width * sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &pStat[rank]);
    cout << "CPU " << rank << " queued2 recv" << endl;
    for (int i = 0; i < rowsPerCPU; i++)
    	Transform1D(hh + i * width, width, HH + i * width);
 
    MPI_Isend(HH, rowsPerCPU * width * sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &pReq[rank]);
    cout << "CPU " << rank << " queued2 send" << endl;
  

  if( rank == 0 )
  {
    for( int cpu = 0; cpu < nCPUs; ++cpu)
    {
      int startRow = cpu * rowsPerCPU;
      MPI_Recv(R + startRow * width, rowsPerCPU * width * sizeof(Complex), MPI_CHAR, cpu, 0, MPI_COMM_WORLD, &pStat[cpu]);
      cout << "CPU " << cpu << " queued2 recv" << endl; 
    }
    cout << "CPU " << rank << " cao" << endl;
 
    Transpose(R, width, height);
    image.SaveImageData("myafter2d.txt", R, width, height);
  }
}

int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  // MPI initialization here
  MPI_Init(&argc, &argv);

  Transform2D(fn.c_str()); // Perform the transform.

  //Transform2DInverse();
    
  // Finalize MPI here
  MPI_Finalize();
}  
  

  
