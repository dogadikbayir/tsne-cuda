// TODO: add copyright

/*
    Compute unnormalized attractive force for barnes-hut approximation of t-SNE.

    Attractive force is given by pij*qij.
*/

#include "kernels/attr_forces.h"
#include <chrono>
#define START_IL_TIMER() start = std::chrono::high_resolution_clock::now();
#define END_IL_TIMER(x) stop = std::chrono::high_resolution_clock::now(); duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); x += duration;

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed with error (%d) at line %d\n",             \
               status, __LINE__);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


__global__
void ComputePijxQijKernel(
                            volatile float * __restrict__ pijqij,
                            const float * __restrict__ pij,
                            const float * __restrict__ points,
                            const int * __restrict__ coo_indices,
                            const int num_points,
                            const int num_nonzero)
{
    register int TID, i, j;
    register float ix, iy, jx, jy, dx, dy;
    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= num_nonzero) return;
    i = coo_indices[2*TID];
    j = coo_indices[2*TID+1];

    ix = points[i]; iy = points[num_points + i];
    jx = points[j]; jy = points[num_points + j];
    dx = ix - jx;
    dy = iy - jy;
    pijqij[TID] = pij[TID] / (1 + dx*dx + dy*dy);
    //atomicAdd(attr_forces + i, pijqij * dx);
    //atomicAdd(attr_forces + num_points + i, pijqij * dy);
}
void tsnecuda::ComputeAttractiveForcesBSR(
                    tsnecuda::GpuOptions &gpu_opt,
                    cusparseHandle_t &handle,
                    cusparseMatDescr_t &bsr_descr,
                    thrust::device_vector<float> &attr_forces,
                    thrust::device_vector<float> &pijqij,
                    thrust::device_vector<float> &sparse_pij_device,
                    float *bsrVal,
                    int *bsrRowPtr,
                    int *bsrColInd,
                    thrust::device_vector<int> &coo_indices,
                    thrust::device_vector<float> &points,
                    thrust::device_vector<float> &ones,
                    const int num_points,
                    const int num_nonzero,
                    const int nnzb)
{
   const int BLOCKSIZE = 1024;
    const int NBLOCKS = iDivUp(num_nonzero, BLOCKSIZE);
    
    //START_IL_TIMER();

    ComputePijxQijKernel<<<NBLOCKS, BLOCKSIZE>>>(
                    thrust::raw_pointer_cast(pijqij.data()),
                    thrust::raw_pointer_cast(sparse_pij_device.data()),
                    thrust::raw_pointer_cast(points.data()),
                    thrust::raw_pointer_cast(coo_indices.data()),
                    num_points,
                    num_nonzero);
    GpuErrorCheck(cudaDeviceSynchronize());
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    int mb = (num_points + BLOCKSIZE-1)/BLOCKSIZE;

    const int m = mb * BLOCKSIZE;
    const int ldb = m;
    const int ldc = m;

    cusparseSbsrmm(handle, CUSPARSE_DIRECTION_ROW,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, mb,
        2, mb, nnzb, &alpha, bsr_descr, bsrVal, bsrRowPtr, bsrColInd,
        BLOCKSIZE, thrust::raw_pointer_cast(ones.data()), ldb, &beta,
        thrust::raw_pointer_cast(attr_forces.data()), ldc );
    GpuErrorCheck(cudaDeviceSynchronize());

    //Second Hadamard Prod.
    thrust::transform(attr_forces.begin(), attr_forces.end(), points.begin(),
        attr_forces.begin(), thrust::multiplies<float>());
    GpuErrorCheck(cudaDeviceSynchronize());

    cusparseSbsrmm(handle, CUSPARSE_DIRECTION_ROW,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, mb,
        2, mb, nnzb, &alpha, bsr_descr, bsrVal, bsrRowPtr, bsrColInd,
        BLOCKSIZE, thrust::raw_pointer_cast(points.data()), ldb, &beta,
        thrust::raw_pointer_cast(attr_forces.data()), ldc );
    GpuErrorCheck(cudaDeviceSynchronize());

}

void tsnecuda::ComputeAttractiveForces(
                    tsnecuda::GpuOptions &gpu_opt,
                    cusparseHandle_t &handle,
                    cusparseMatDescr_t &descrSp,
                    thrust::device_vector<float> &attr_forces,
                    thrust::device_vector<float> &pijqij,
                    thrust::device_vector<float> &sparse_pij,
                    thrust::device_vector<int> &pij_row_ptr,
                    thrust::device_vector<int> &pij_col_ind,
                    thrust::device_vector<int> &coo_indices,
                    thrust::device_vector<float> &points,
                    thrust::device_vector<float> &ones,
                    const int num_points,
                    float &time_firstSPDM,
                    float &time_secondSPDM,
                    float &time_mul,
                    float &time_pijkern,
                    const int num_nonzero)
{
    // Computes pij*qij for each i,j
    // TODO: this is bad style
    //
    
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    //init timers
    auto time_pijkern_ = duration;
    auto time_firstSPDM_ = duration;
    auto time_secondSPDM_ = duration;
    auto time_mul_ = duration;

    const int BLOCKSIZE = 1024;
    const int NBLOCKS = iDivUp(num_nonzero, BLOCKSIZE);
    
    START_IL_TIMER();

    ComputePijxQijKernel<<<NBLOCKS, BLOCKSIZE>>>(
                    thrust::raw_pointer_cast(pijqij.data()),
                    thrust::raw_pointer_cast(sparse_pij.data()),
                    thrust::raw_pointer_cast(points.data()),
                    thrust::raw_pointer_cast(coo_indices.data()),
                    num_points,
                    num_nonzero);
    GpuErrorCheck(cudaDeviceSynchronize());
    
    END_IL_TIMER(time_pijkern_);
    //size_t bufferSize = 0;
    //void* dBuffer = NULL;
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    START_IL_TIMER();
    // (PijxQij)*(Ones)
    cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          num_points, 2, num_points, num_nonzero, &alpha, descrSp,
          thrust::raw_pointer_cast(pijqij.data()),
          thrust::raw_pointer_cast(pij_row_ptr.data()),
          thrust::raw_pointer_cast(pij_col_ind.data()),
          thrust::raw_pointer_cast(ones.data()), num_points, &beta,
          thrust::raw_pointer_cast(attr_forces.data()), num_points);

    GpuErrorCheck(cudaDeviceSynchronize());
    END_IL_TIMER(time_firstSPDM_);
    // The first Hadamard product
    START_IL_TIMER(); 
    thrust::transform(attr_forces.begin(), attr_forces.end(), points.begin(),
        attr_forces.begin(), thrust::multiplies<float>());

    END_IL_TIMER(time_mul_);
    alpha = -1.0f;
    beta = 1.0f;
    
    START_IL_TIMER();
    // (PijxQij)*Y
    cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          num_points, 2, num_points, num_nonzero, &alpha, descrSp,
          thrust::raw_pointer_cast(pijqij.data()),
          thrust::raw_pointer_cast(pij_row_ptr.data()),
          thrust::raw_pointer_cast(pij_col_ind.data()),
          thrust::raw_pointer_cast(points.data()), num_points, &beta,
          thrust::raw_pointer_cast(attr_forces.data()), num_points);

    END_IL_TIMER(time_secondSPDM_);

    time_firstSPDM = ((float) time_firstSPDM_.count()) / 1000000.0;
    time_secondSPDM = ((float) time_secondSPDM_.count()) / 1000000.0;  
    time_mul = ((float) time_mul_.count()) / 1000000.0; 
    time_pijkern = ((float) time_pijkern_.count()) / 1000000.0; 
}
