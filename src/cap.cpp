#include <stdio.h>

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
    
//	cudaStreamSynchronize(stream1);
//	cudaStreamSynchronize(stream2);
    cudaMemcpyAsync(a, d_a, SIZE*sizeof(int), cudaMemcpyDeviceToHost, get_cuda_stream(1));
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
