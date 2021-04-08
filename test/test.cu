#include "nufft.cuh"
#include <iostream>
#include <thrust/complex.h>
using namespace thrust;
using namespace std::complex_literals;
//#include <cstdlib>

int main(){
    float * x = (float*)malloc(sizeof(float)*100);
    complex<float> *c = (complex<float> *)malloc(sizeof(complex<float>)*100);
    srand(1221);
    for(int i=0; i<100; i++){
        x[i] = rand() / float(RAND_MAX) * 2000;
        c[i] = exp(1.0if*x[i]);
    }
    directFT_1d( 51,  c, x, 100, 1);

    return 0;
}

