#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#include "cap.h"
#include "darknet.h"

#define SIZE 1024
//#define SIZE 1108992
 
static cudaStream_t streamsArray[16];    // cudaStreamSynchronize( get_cuda_stream() );
static int streamInit[16] = { 0 };

cudaStream_t get_cuda_stream(int stream_index) {
    int i = stream_index;
	int leastPriority=-1, greatestPriority=-1;
    if (!streamInit[i]) {
        cudaError_t status = cudaStreamCreate(&streamsArray[i]);
		cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

        if (status != cudaSuccess) {
            printf(" cudaStreamCreate error: %d \n", status);
            const char *s = cudaGetErrorString(status);
            printf("CUDA Error: %s\n", s);
            status = cudaStreamCreateWithFlags(&streamsArray[i], cudaStreamDefault);

//			check_error(status);
		}
        streamInit[i] = 1;
    }
    return streamsArray[i];
}

__global__ void VectorAdd(int *a, int *b, int *c, int n){
//	__shared__ int s[12240];
    int i = threadIdx.x;
 
//    printf("threadIdx.x : %d, n : %d\n", i, n);
    
    for (i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void vector_add_gpu()
{
    printf("vector add gpu start \n");
    int *a, *b, *c, *d, *e, *f, *g, *h, *i;
    int *d_a, *d_b, *d_c, *d_d, *d_e, *d_f, *d_g, *d_h, *d_i;

    a = (int *)malloc(SIZE*sizeof(int));
    b = (int *)malloc(SIZE*sizeof(int));
    c = (int *)malloc(SIZE*sizeof(int));
	d = (int *)malloc(SIZE*sizeof(int));
	e = (int *)malloc(SIZE*sizeof(int));
    f = (int *)malloc(SIZE*sizeof(int));
    g = (int *)malloc(SIZE*sizeof(int));
    h = (int *)malloc(SIZE*sizeof(int));
    i = (int *)malloc(SIZE*sizeof(int));
 
    cudaMalloc(&d_a, SIZE*sizeof(int));
    cudaMalloc(&d_b, SIZE*sizeof(int));
    cudaMalloc(&d_c, SIZE*sizeof(int));
    cudaMalloc(&d_d, SIZE*sizeof(int));
    cudaMalloc(&d_e, SIZE*sizeof(int));
    cudaMalloc(&d_f, SIZE*sizeof(int));
    cudaMalloc(&d_g, SIZE*sizeof(int));
    cudaMalloc(&d_h, SIZE*sizeof(int));
    cudaMalloc(&d_i, SIZE*sizeof(int));

    for (int k = 0; k<1024; ++k)
    {
        a[k] = k;
        b[k] = k;
        c[k] = 0;
        d[k] = k;
        e[k] = k;
        f[k] = 0;
		g[k] = k;
		h[k] = k;
		i[k] = 0;
    }
    
    cudaMemcpyAsync(d_a, a, SIZE*sizeof(int), cudaMemcpyHostToDevice, get_cuda_stream(1));
    cudaMemcpyAsync(d_b, b, SIZE*sizeof(int), cudaMemcpyHostToDevice, get_cuda_stream(1));
    cudaMemcpyAsync(d_c, c, SIZE*sizeof(int), cudaMemcpyHostToDevice, get_cuda_stream(1));
    cudaMemcpyAsync(d_d, d, SIZE*sizeof(int), cudaMemcpyHostToDevice, get_cuda_stream(1));
    cudaMemcpyAsync(d_e, e, SIZE*sizeof(int), cudaMemcpyHostToDevice, get_cuda_stream(1));
    cudaMemcpyAsync(d_f, f, SIZE*sizeof(int), cudaMemcpyHostToDevice, get_cuda_stream(1));
    cudaMemcpyAsync(d_g, d, SIZE*sizeof(int), cudaMemcpyHostToDevice, get_cuda_stream(1));
    cudaMemcpyAsync(d_h, e, SIZE*sizeof(int), cudaMemcpyHostToDevice, get_cuda_stream(1));
    cudaMemcpyAsync(d_i, f, SIZE*sizeof(int), cudaMemcpyHostToDevice, get_cuda_stream(1));

    //for (int i = 0; i < 100; i++){
    VectorAdd << < 1024, 512,0, get_cuda_stream(1)>> >(d_d, d_e, d_f, 1024);
    VectorAdd << < 1024, 512,0, get_cuda_stream(1)>> >(d_d, d_e, d_f, 1024);
    VectorAdd << < 1024, 512,0, get_cuda_stream(1)>> >(d_d, d_e, d_f, 1024);
    VectorAdd << < 1024, 512,0, get_cuda_stream(1)>> >(d_d, d_e, d_f, 1024);
    VectorAdd << < 1024, 512,0, get_cuda_stream(1)>> >(d_d, d_e, d_f, 1024);

//    VectorAdd << < 1024, 512>> >(d_d, d_e, d_f, 1024);
//    VectorAdd << < 1024, 512>> >(d_d, d_e, d_f, 1024);
//    VectorAdd << < 1024, 512>> >(d_d, d_e, d_f, 1024);
//    VectorAdd << < 1024, 512>> >(d_d, d_e, d_f, 1024);
//    VectorAdd << < 1024, 512>> >(d_d, d_e, d_f, 1024);
	//}
    
	cudaStreamSynchronize(get_cuda_stream(1));
    printf("Async start \n");
    cudaMemcpyAsync(a, d_a, SIZE*sizeof(int), cudaMemcpyDeviceToHost, get_cuda_stream(1));
    printf("Async end \n");
    printf("Sync end \n");
//    cudaMemcpyAsync(b, d_b, SIZE*sizeof(int), cudaMemcpyDeviceToHost, get_cuda_stream(1));
//    cudaMemcpyAsync(c, d_c, SIZE*sizeof(int), cudaMemcpyDeviceToHost, get_cuda_stream(1));
//	//cudaStreamDestroy(stream1);
//    cudaMemcpyAsync(d, d_d, SIZE*sizeof(int), cudaMemcpyDeviceToHost, get_cuda_stream(1));
//    cudaMemcpyAsync(e, d_e, SIZE*sizeof(int), cudaMemcpyDeviceToHost, get_cuda_stream(1));
//    cudaMemcpyAsync(f, d_f, SIZE*sizeof(int), cudaMemcpyDeviceToHost, get_cuda_stream(1));
//	cudaMemcpyAsync(d, d_g, SIZE*sizeof(int), cudaMemcpyDeviceToHost, get_cuda_stream(1));
//	cudaMemcpyAsync(e, d_h, SIZE*sizeof(int), cudaMemcpyDeviceToHost, get_cuda_stream(1));
//    cudaMemcpyAsync(f, d_i, SIZE*sizeof(int), cudaMemcpyDeviceToHost, get_cuda_stream(1));
 
    free(a);
    free(b);
    free(c);
    free(d);
    free(e);
    free(f);
 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_e);
    cudaFree(d_f);
    printf("vector add gpu end \n");
}

//int main(){
//    int *a, *b, *c, *d, *e, *f, *g, *h, *i;
//    int *d_a, *d_b, *d_c, *d_d, *d_e, *d_f, *d_g, *d_h, *d_i;
////	cudaStream_t stream1, stream2;
// 
//    // 호스트의 메모리에 할당한다.
//    a = (int *)malloc(SIZE*sizeof(int));
//    b = (int *)malloc(SIZE*sizeof(int));
//    c = (int *)malloc(SIZE*sizeof(int));
//	d = (int *)malloc(SIZE*sizeof(int));
//	e = (int *)malloc(SIZE*sizeof(int));
//    f = (int *)malloc(SIZE*sizeof(int));
//    g = (int *)malloc(SIZE*sizeof(int));
//    h = (int *)malloc(SIZE*sizeof(int));
//    i = (int *)malloc(SIZE*sizeof(int));
//    
//    // cudaMalloc(destination, number of byte)로 device의 메모리를 할당한다.
//    cudaMalloc(&d_a, SIZE*sizeof(int));
//    cudaMalloc(&d_b, SIZE*sizeof(int));
//    cudaMalloc(&d_c, SIZE*sizeof(int));
//    cudaMalloc(&d_d, SIZE*sizeof(int));
//    cudaMalloc(&d_e, SIZE*sizeof(int));
//    cudaMalloc(&d_f, SIZE*sizeof(int));
//    cudaMalloc(&d_g, SIZE*sizeof(int));
//    cudaMalloc(&d_h, SIZE*sizeof(int));
//    cudaMalloc(&d_i, SIZE*sizeof(int));
//    
//    // 초기화
//    for (int k = 0; k<1024; ++k)
//    {
//        a[k] = k;
//        b[k] = k;
//        c[k] = 0;
//        d[k] = k;
//        e[k] = k;
//        f[k] = 0;
//		g[k] = k;
//		h[k] = k;
//		i[k] = 0;
//    }
//
////	cudaStreamCreate(&stream1);
////	cudaStreamCreate(&stream2);
//    
//    // cudaMemcpy(destination, source, number of byte, cudaMemcpyHostToDevice)로 호스트에서 디바이스로 메모리를 카피한다.
//    cudaMemcpy(d_a, a, SIZE*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_b, b, SIZE*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_c, c, SIZE*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_d, d, SIZE*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_e, e, SIZE*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_f, f, SIZE*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_g, d, SIZE*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_h, e, SIZE*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_i, f, SIZE*sizeof(int), cudaMemcpyHostToDevice);
//
// 
// 
//    // 함수 호출을 위해서 새로운 신텍스 요소를 추가할 필요가 있다.
//    // 첫번째 parameter는 블럭의 수이다. 예제에서는 스레드 블럭이 하나이다.
//    // SIZE는 1024개의 스레드를 의미한다.
//	//for (int i = 0; i < 100; i++){
//	for (;;){
//		//VectorAdd << < 1, SIZE>> >(d_a, d_b, d_c, SIZE);
//		//VectorAdd << < 2*16, 512,0, get_cuda_stream(0)>> >(d_a, d_b, d_c, 1024);
////		cudaStreamSynchronize(get_cuda_stream(0));
//		VectorAdd << < 24, 512,0, get_cuda_stream(1)>> >(d_d, d_e, d_f, 1024);
////		cudaStreamSynchronize(get_cuda_stream(1));
//		//VectorAdd << < 1, 1024,0, get_cuda_stream(2)>> >(d_g, d_h, d_i, 1024);
//	}
//    //fill_kernel <<<23514,512,0,stream1>>>(100,0,100,1);
//    //fill_kernel <<<23514,512,0,stream1>>>(100,0,100,1);
//    
//    //cudaMemcpy(source, destination, number of byte, cudaMemDeviceToHost)로 디바이스의 메모리(연산 결과 데이터)를 호스트에 카피한다.
////	cudaStreamSynchronize(stream1);
////	cudaStreamSynchronize(stream2);
//    cudaMemcpy(a, d_a, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(b, d_b, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(c, d_c, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
//	//cudaStreamDestroy(stream1);
//    cudaMemcpy(d, d_d, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(e, d_e, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(f, d_f, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
//	cudaMemcpy(d, d_g, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
//	cudaMemcpy(e, d_h, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(f, d_i, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
//
//	//cudaStreamDestroy(stream2);
// 
////    for (int i = 0; i<1024; ++i){
////        printf("c[%d] = %d\n", i, c[i]);
////        printf("f[%d] = %d\n", i, f[i]);
////	}
// 
//    // 호스트의 메모리 할당 해제
//    free(a);
//    free(b);
//    free(c);
//    free(d);
//    free(e);
//    free(f);
// 
//    
//    // cudaFree(d_a)를 통해 디바이스의 메모리를 할당 해제
//    cudaFree(d_a);
//    cudaFree(d_b);
//    cudaFree(d_c);
//    cudaFree(d_d);
//    cudaFree(d_e);
//    cudaFree(d_f);
// 
//    return 0;
//}
