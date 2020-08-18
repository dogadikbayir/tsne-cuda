/*
    Compute t-SNE via Barnes-Hut for NlogN time.
*/

#include "include/fit_tsne.h"
#include <chrono>
#include <string>

#define START_IL_REORDER() startReorder = std::chrono::high_resolution_clock::now();
#define END_IL_REORDER(x) endReorder = std::chrono::high_resolution_clock::now(); duration = std::chrono::duration_cast<std::chrono::microseconds>(endReorder-startReorder); x += duration; total_time += duration;
#define START_IL_TIMER() start = std::chrono::high_resolution_clock::now();
#define END_IL_TIMER(x) stop = std::chrono::high_resolution_clock::now(); duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); x += duration; total_time += duration;
#define PRINT_IL_TIMER(x) std::cout << #x << ": " << ((float) x.count()) / 1000000.0 << "s" << std::endl

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed with error (%d) at line %d\n",             \
               status, __LINE__);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
using rabbit_order::vint;

//typedef std::vector<std::vector<std::pair<vint, float>>> adjacency_list;
//typedef std::tuple<vint, vint, float> edge;

std::vector<edge_list::edge> tsnecuda::gen_edgelist(int *row_ptr, int *col_ind, int num_points, int num_nnz) {
  //Init empty vector
  std::vector<edge_list::edge> edges;

  //Iterate over col indices
  int row_ind = 0;
  int counter = 0;
  int num_nnz_row = row_ptr[1];
  int temp_counter = 0;
  edge_list::edge e;
  while(counter < num_nnz) {
    if (temp_counter < num_nnz_row) {
      e = edge_list::edge {row_ind, col_ind[counter], 1.0};
      edges.push_back(e);
      counter++;
      temp_counter++;
    }
    else { temp_counter = 0; row_ind++;num_nnz_row = row_ptr[row_ind + 1] - row_ptr[row_ind];}
  }
  //assert(edges != NULL);
  assert(edges.size() == num_nnz);

  return edges;
}

void tsnecuda::dump_edges(std::string filename, std::vector<edge_list::edge> edges){
  std::ofstream dump_edges;
  dump_edges.open("dump_edges.out");
  
  for(auto edge : edges) {
    dump_edges << std::get<0>(edge) << " " << std::get<1>(edge) << std::endl;
  }
  
  dump_edges.close();
  
}
template<typename RandomAccessRange>
tsnecuda::adjacency_list tsnecuda::make_adj_list(const vint n, const RandomAccessRange& es) {
  using std::get;

  // Symmetrize the edge list and remove self-loops simultaneously
  std::vector<edge_list::edge> ss(boost::size(es) * 2);
  #pragma omp parallel for
  for (size_t i = 0; i < boost::size(es); ++i) {
    auto& e = es[i];
    if (get<0>(e) != get<1>(e)) {
      ss[i * 2    ] = std::make_tuple(get<0>(e), get<1>(e), get<2>(e));
      ss[i * 2 + 1] = std::make_tuple(get<1>(e), get<0>(e), get<2>(e));
    } else {
      // Insert zero-weight edges instead of loops; they are ignored in making
      // an adjacency list
      ss[i * 2    ] = std::make_tuple(0, 0, 0.0f);
      ss[i * 2 + 1] = std::make_tuple(0, 0, 0.0f);
    }
  }

  // Sort the edges
  __gnu_parallel::sort(ss.begin(), ss.end());

  // Convert to an adjacency list
  adjacency_list adj(n);
  #pragma omp parallel
  {
    // Advance iterators to a boundary of a source vertex
    const auto adv = [](auto it, const auto first, const auto last) {
      while (first != it && it != last && get<0>(*(it - 1)) == get<0>(*it))
        ++it;
      return it;
    };

    // Compute an iterator range assigned to this thread
    const int    p      = omp_get_max_threads();
    const size_t t      = static_cast<size_t>(omp_get_thread_num());
    const size_t ifirst = ss.size() / p * (t)   + std::min(t,   ss.size() % p);
    const size_t ilast  = ss.size() / p * (t+1) + std::min(t+1, ss.size() % p);
    auto         it     = adv(ss.begin() + ifirst, ss.begin(), ss.end());
    const auto   last   = adv(ss.begin() + ilast,  ss.begin(), ss.end());

    // Reduce edges and store them in std::vector
    while (it != last) {
      const vint s = get<0>(*it);

      // Obtain an upper bound of degree and reserve memory
      const auto maxdeg = 
          std::find_if(it, last, [s](auto& x) {return get<0>(x) != s;}) - it;
      adj[s].reserve(maxdeg);

      while (it != last && get<0>(*it) == s) {
        const vint t = get<1>(*it);
        float      w = 0.0;
        while (it != last && get<0>(*it) == s && get<1>(*it) == t)
          w += get<2>(*it++);
        if (w > 0.0)
          adj[s].push_back({t, w});
      }

      // The actual degree can be smaller than the upper bound
      adj[s].shrink_to_fit();
    }
  }

  return adj;
}

//Custom comparator for permuting a thrust vector
struct copy_idx_func : public thrust::unary_function<unsigned, unsigned>
{
  size_t c;
  unsigned *p;
  copy_idx_func(const size_t _c, unsigned *_p) : c(_c), p(_p) {};
  __host__ __device__
    unsigned operator()(unsigned idx) {
      unsigned myrow = idx/c;
      unsigned newrow = p[myrow] - 1;
      unsigned mycol = idx%c;
      return newrow*c+mycol;
    }
};
//Save CPU array to file
template <typename T>
void tsnecuda::save_cpu(std::string filename, T * host_arr, int size) {
    std::ofstream dump_file;
    dump_file.open(filename + std::to_string(size));
    for(int i=0;i<size;i++){
      dump_file << host_arr[i] << std::endl;
    }
    dump_file.close();
}
//Save GPU array to file
template <typename T>
void tsnecuda::save_coo(std::string filename, thrust::device_vector<T> device_vec, int size_coo ) {
    std::cout << "Entered save_coo" << std::endl;
    std::ofstream dump_coo;
    T *h_coo = (T *)malloc((size_coo * 2)*sizeof(T));
    cudaMemcpy(h_coo, thrust::raw_pointer_cast(device_vec.data()), sizeof(T)*(size_coo*2), cudaMemcpyDeviceToHost);
    std::cout << "Allocated memory" << std::endl;
    save_cpu(filename, h_coo, size_coo*2);
    
}

//Split string
std::vector<std::string> tsnecuda::split (std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}
void tsnecuda::RunTsne(tsnecuda::Options &opt,
                       tsnecuda::GpuOptions &gpu_opt)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto endReorder = std::chrono::high_resolution_clock::now();
    auto startReorder = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    auto total_time = duration;
    auto _time_initialization = duration;
    auto _time_knn = duration;
    auto _time_knn2 = duration;
    auto _time_normknn = duration;
    auto _time_symmetry = duration;
    auto _time_perm = duration;
    auto _time_reorder = duration;
    auto _time_reord_buff = duration;
    auto _time_mapping = duration;
    auto _time_devicecopy = duration;
    auto _time_hostcopy = duration;
    auto _time_tot_perm = duration;
    auto _time_init_low_dim = duration;
    auto _time_init_fft = duration;
    auto _time_precompute_2d = duration;
    auto _time_nbodyfft = duration;
    auto _time_compute_charges = duration;
    auto _time_other = duration;
    auto _time_norm = duration;
    auto _time_attr = duration;
    auto _time_apply_forces = duration;
    auto _time_map_points = duration;

    // Check the validity of the options file
    if (!opt.validate()) {
        std::cout << "E: Invalid options file. Terminating." << std::endl;
        return;
    }
	std::cout << "Start" << std::endl;
    START_IL_TIMER();

    if (opt.verbosity > 0) {
        std::cout << "Initializing cuda handles... " << std::flush;
    }

    // Construct the handles
    cublasHandle_t dense_handle;
    CublasSafeCall(cublasCreate(&dense_handle));
    cusparseHandle_t sparse_handle;
    CusparseSafeCall(cusparseCreate(&sparse_handle));
    std::cout << "Created cublas handle" << std::endl;
    // Set CUDA device properties
    const int num_blocks = gpu_opt.sm_count;

    // Construct sparse matrix descriptor
    cusparseMatDescr_t sparse_matrix_descriptor;

    //cusparseDnMatDescr_t dense_pts;
    //cusparseDnMatDescr_t dense_pijqij;

    cusparseCreateMatDescr(&sparse_matrix_descriptor);
        
    
    cusparseSetMatType(sparse_matrix_descriptor, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(sparse_matrix_descriptor,CUSPARSE_INDEX_BASE_ZERO);

    // Setup some return information if we're working on snapshots
    int snap_num = 0;
    int snap_interval = 1;
    if (opt.return_style == tsnecuda::RETURN_STYLE::SNAPSHOT) {
        snap_interval = opt.iterations / (opt.num_snapshots - 1);
    }

    // Get constants from options
    const int num_points = opt.num_points;
    const int num_neighbors = (opt.num_neighbors < num_points) ? opt.num_neighbors : num_points;
    const float *high_dim_points = opt.points;
    const int high_dim = opt.num_dims;
    const float perplexity = opt.perplexity;
    const float perplexity_search_epsilon = opt.perplexity_search_epsilon;
    const float eta = opt.learning_rate;
    float momentum = opt.pre_exaggeration_momentum;
    float attr_exaggeration = opt.early_exaggeration;
    float normalization;

    // Allocate host memory
    float *knn_squared_distances = new float[num_points * num_neighbors];
    memset(knn_squared_distances, 0, num_points * num_neighbors * sizeof(float));
    long *knn_indices = new long[num_points * num_neighbors];

    // Set cache configs
    // cudaFuncSetCacheConfig(tsnecuda::IntegrationKernel, cudaFuncCachePreferL1);
    // cudaFuncSetCacheConfig(tsnecuda::ComputePijxQijKernel, cudaFuncCachePreferShared);
    GpuErrorCheck(cudaDeviceSynchronize());


    END_IL_TIMER(_time_initialization);
    START_IL_TIMER();

    if (opt.verbosity > 0) {
        std::cout << "done.\nKNN Computation... " << std::flush;
    }
    // Compute approximate K Nearest Neighbors and squared distances
    tsnecuda::util::KNearestNeighbors(gpu_opt, knn_indices, knn_squared_distances, high_dim_points, high_dim, num_points, num_neighbors);
    END_IL_TIMER(_time_knn);
    START_IL_TIMER();
    thrust::device_vector<long> knn_indices_long_device(knn_indices, knn_indices + num_points * num_neighbors);
    thrust::device_vector<int> knn_indices_device(num_points * num_neighbors);
    tsnecuda::util::PostprocessNeighborIndices(gpu_opt, knn_indices_device, knn_indices_long_device,
                                                        num_points, num_neighbors);
    END_IL_TIMER(_time_knn2);
    START_IL_TIMER();
    // Max-norm the distances to avoid exponentiating by large numbers
    thrust::device_vector<float> knn_squared_distances_device(knn_squared_distances,
                                            knn_squared_distances + (num_points * num_neighbors));
    tsnecuda::util::MaxNormalizeDeviceVector(knn_squared_distances_device);

    END_IL_TIMER(_time_normknn);
    START_IL_TIMER();

    if (opt.verbosity > 0) {
        std::cout << "done.\nComputing Pij matrix... " << std::flush;
    }

    // Search Perplexity
    thrust::device_vector<float> pij_non_symmetric_device(num_points * num_neighbors);
    tsnecuda::SearchPerplexity(gpu_opt, dense_handle, pij_non_symmetric_device, knn_squared_distances_device,
                                    perplexity, perplexity_search_epsilon, num_points, num_neighbors);

    // Clean up memory
    knn_squared_distances_device.clear();
    knn_squared_distances_device.shrink_to_fit();
    knn_indices_long_device.clear();
    knn_indices_long_device.shrink_to_fit();
    delete[] knn_squared_distances;

    if(opt.reorder == 7){
      //Dump knn info to file
      std::ofstream knn_file;
      knn_file.open("knn_" + std::to_string(opt.num_points));
      //host_ys = new float[num_points * 2];
      //dump_file << num_points << " " << 2 << std::endl;
      for(int i=0; i<num_points*num_neighbors; i++){
        knn_file << knn_indices[i] << " ";
      }

    }
    delete[] knn_indices;

    // Symmetrize the pij matrix
    thrust::device_vector<float> sparse_pij_device;
    thrust::device_vector<int> pij_row_ptr_device;
    thrust::device_vector<int> pij_col_ind_device;
    tsnecuda::util::SymmetrizeMatrix(sparse_handle, sparse_pij_device, pij_row_ptr_device,
                                        pij_col_ind_device, pij_non_symmetric_device, knn_indices_device,
                                        opt.magnitude_factor, num_points, num_neighbors);

    const int num_nonzero = sparse_pij_device.size();
        // Clean up memory
    knn_indices_device.clear();
    knn_indices_device.shrink_to_fit();
    pij_non_symmetric_device.clear();
    pij_non_symmetric_device.shrink_to_fit();

    // Declare memory

    //thrust::device_vector<float> pijqij(sparse_pij_device.size());
    thrust::device_vector<float> repulsive_forces_device(opt.num_points * 2, 0);
    thrust::device_vector<float> attractive_forces_device(opt.num_points * 2, 0);
    thrust::device_vector<float> gains_device(opt.num_points * 2, 1);
    thrust::device_vector<float> old_forces_device(opt.num_points * 2, 0); // for momentum
    thrust::device_vector<float> normalization_vec_device(opt.num_points);
    thrust::device_vector<float> ones_device(opt.num_points * 2, 1); // This is for reduce summing, etc.
    thrust::device_vector<int> coo_indices_device(sparse_pij_device.size()*2);

    //tsnecuda::util::Csr2Coo(gpu_opt, coo_indices_device, pij_row_ptr_device,
      //                      pij_col_ind_device, num_points, num_nonzero);

    END_IL_TIMER(_time_symmetry);
    START_IL_TIMER();
    
        
    
    

    if (opt.verbosity > 0) {
        std::cout << "done.\nInitializing low dim points... " << std::flush;
        //std::ifstream re_pij;
        //re_pij.open("./re_pij.txt");

    }

    // Initialize Low-Dim Points
    thrust::device_vector<float> points_device(num_points * 2);
    thrust::device_vector<float> random_vector_device(points_device.size());

    std::default_random_engine generator(opt.random_seed);
    std::normal_distribution<float> distribution1(0.0, 1.0);
    thrust::host_vector<float> h_points_device(num_points * 2);

    
    // Initialize random noise vector
    for (int i = 0; i < h_points_device.size(); i++) h_points_device[i] = 0.001 * distribution1(generator);
    thrust::copy(h_points_device.begin(), h_points_device.end(), random_vector_device.begin());

    // TODO: this will only work with gaussian init
    if (opt.initialization == tsnecuda::TSNE_INIT::UNIFORM) { // Random uniform initialization
        points_device = tsnecuda::util::RandomDeviceVectorInRange(generator, points_device.size(), -5, 5);
    } else if (opt.initialization == tsnecuda::TSNE_INIT::GAUSSIAN) { // Random gaussian initialization
        // Generate some Gaussian noise for the points
        for (int i = 0; i < h_points_device.size(); i++) h_points_device[i] = 0.0001 * distribution1(generator);
        thrust::copy(h_points_device.begin(), h_points_device.end(), points_device.begin());
    } else if (opt.initialization == tsnecuda::TSNE_INIT::RESUME) { // Preinit from vector
        // Load from vector
        if(opt.preinit_data != nullptr) {
          thrust::copy(opt.preinit_data, opt.preinit_data + points_device.size(), points_device.begin());
        } else {
          std::cerr << "E: Invalid initialization. Initialization points are null." << std::endl;
          exit(1);
        }
    } else if (opt.initialization == tsnecuda::TSNE_INIT::VECTOR) { // Preinit from vector points only
        // Copy the pre-init data
        if(opt.preinit_data != nullptr) {
          thrust::copy(opt.preinit_data, opt.preinit_data + points_device.size(), points_device.begin());
        } else {
          std::cerr << "E: Invalid initialization. Initialization points are null." << std::endl;
          exit(1);
        }
    } else { // Invalid initialization
        std::cerr << "E: Invalid initialization type specified." << std::endl;
        exit(1);
    }

    END_IL_TIMER(_time_init_low_dim);
    START_IL_TIMER();

    if (opt.verbosity > 0) {
        std::cout << "done.\nInitializing CUDA memory... " << std::flush;
    }
    
    //Define the cuSparse matrices
    //
    //Create the cuSparse matrix (CSR)
    //cusparseCreateMatDescr(&sparse_matrix_descriptor);
    //cusparseSetMatType(sparse_matrix_descriptor, CUSPARSE_MATRIX_TYPE_GENERAL);
    //cusparseSetMatIndexBase(sparse_matrix_descriptor,
      //  CUSPARSE_INDEX_BASE_ZERO);
    
    
      float *h_pij_vals2 = (float *)malloc((num_nonzero)*sizeof(float));
      //h_pij_vals = thrust::raw_pointer_cast(sparse_pij_device.data());
      cudaMemcpy(h_pij_vals2, thrust::raw_pointer_cast(sparse_pij_device.data()), sizeof(float)*(num_nonzero), cudaMemcpyDeviceToHost);
     
      int *h_pij_row_ptr2 = (int *)malloc((num_points+1)*sizeof(int));
      //h_pij_row_ptr = thrust::raw_pointer_cast(pij_row_ptr_device.data());
      cudaMemcpy(h_pij_row_ptr2, thrust::raw_pointer_cast(pij_row_ptr_device.data()), sizeof(int)*(num_points+1), cudaMemcpyDeviceToHost);

      int *h_pij_col_ind2 = (int *)malloc((num_nonzero)*sizeof(int));
      //h_pij_col_ind = thrust::raw_pointer_cast(pij_col_ind_device.data());
      cudaMemcpy(h_pij_col_ind2, thrust::raw_pointer_cast(pij_col_ind_device.data()), sizeof(int)*(num_nonzero), cudaMemcpyDeviceToHost);

    std::ofstream vals_file;
    std::ofstream row_file;
    std::ofstream ind_file;
    //vals_file.open("vals_" + std::to_string(opt.num_points));
    //row_file.open("rows_" + std::to_string(opt.num_points));
    //ind_file.open("ind_" + std::to_string(opt.num_points));
/*
      for(int i=0; i<num_nonzero; i++){
        vals_file << h_pij_vals2[i] << " ";
      }
      for(int i=0; i<num_points+1; i++){
        row_file << h_pij_row_ptr2[i] << " ";
      }
      for(int i=0; i<num_nonzero; i++){
        ind_file << h_pij_col_ind2[i] << " ";
      }
*/

    //permute the pij sparse matrix
    std::cout << "Num num_nonzero: " << num_nonzero << std::endl;
    tsnecuda::util::Csr2Coo(gpu_opt, coo_indices_device, pij_row_ptr_device, pij_col_ind_device, num_points, num_nonzero);
    std::cout << "just after first csr2coo..." << std::endl;
    //tsnecuda::save_coo("coo_before_", coo_indices_device, num_nonzero);
    std::cout << "just after savecoo" << std::endl;
    //START_IL_REORDER();
//======
//============================================================================================================== 
    //RCM Reorder
    if(opt.reorder==1) {
      std::cout << "Entered RCM branch" << std::endl;
      START_IL_TIMER();
      int issym = 0;
      int *h_Q = NULL;
      //int *h_pij_row_ptr_b = NULL;
      int *h_mapBfromA = NULL;
      //float *h_pij_vals_b = NULL;
      //int *h_pij_col_ind_b = NULL;      
      cusolverSpHandle_t sol_handle = NULL;
      checkCudaErrors(cusolverSpCreate(&sol_handle));
       std::cout << "Created sparse solver handle" << std::endl;
      float *h_pij_vals = (float *)malloc((num_nonzero)*sizeof(float));
      //h_pij_vals = thrust::raw_pointer_cast(sparse_pij_device.data());
      cudaMemcpy(h_pij_vals, thrust::raw_pointer_cast(sparse_pij_device.data()), sizeof(float)*(num_nonzero), cudaMemcpyDeviceToHost);
     
      int *h_pij_row_ptr = (int *)malloc((num_points+1)*sizeof(int));
      //h_pij_row_ptr = thrust::raw_pointer_cast(pij_row_ptr_device.data());
      cudaMemcpy(h_pij_row_ptr, thrust::raw_pointer_cast(pij_row_ptr_device.data()), sizeof(int)*(num_points+1), cudaMemcpyDeviceToHost);

      int *h_pij_col_ind = (int *)malloc((num_nonzero)*sizeof(int));
      //h_pij_col_ind = thrust::raw_pointer_cast(pij_col_ind_device.data());
      cudaMemcpy(h_pij_col_ind, thrust::raw_pointer_cast(pij_col_ind_device.data()), sizeof(int)*(num_nonzero), cudaMemcpyDeviceToHost);

      h_Q = (int *)malloc(sizeof(int)*num_points);
      h_mapBfromA = (int *)malloc(sizeof(int)*num_nonzero);
      
      //check if memory has been allocated without any issues
      assert(NULL != h_Q);
      assert(NULL != h_mapBfromA);
      
      std::cout << "Assertion done" << std::endl;
      END_IL_TIMER(_time_hostcopy);      
      //Compute the permutation vector
      std::cout << "Permuting matrix...";
      START_IL_TIMER();
      if(opt.reopt == 0) {                // RCM
        checkCudaErrors(cusolverSpXcsrsymrcmHost(sol_handle, num_points, num_nonzero, sparse_matrix_descriptor, h_pij_row_ptr, h_pij_col_ind, h_Q));

      }
      else{
        checkCudaErrors(cusolverSpXcsrsymamdHost(sol_handle, num_points, num_nonzero, sparse_matrix_descriptor, h_pij_row_ptr, h_pij_col_ind, h_Q));
        
      }
      END_IL_TIMER(_time_perm);
      std::cout << "Permutation computed..." << std::endl;
      
           
      size_t size_perm = 0;
      void *buffer_cpu = NULL;
      START_IL_TIMER();
      checkCudaErrors(cusolverSpXcsrperm_bufferSizeHost(sol_handle, num_points, num_points, num_nonzero, sparse_matrix_descriptor, h_pij_row_ptr, h_pij_col_ind, h_Q, h_Q, &size_perm));
      END_IL_TIMER(_time_reord_buff);
      buffer_cpu = (void*)malloc(sizeof(char)*size_perm);
      assert(NULL!=buffer_cpu);


      for(int j = 0 ; j < num_nonzero ; j++)
      {
        h_mapBfromA[j] = j;
      }
      START_IL_TIMER();
      checkCudaErrors(cusolverSpXcsrpermHost(sol_handle, num_points, num_points, num_nonzero ,sparse_matrix_descriptor, h_pij_row_ptr, h_pij_col_ind, h_Q, h_Q, h_mapBfromA, buffer_cpu));
      END_IL_TIMER(_time_reorder);
      //Map the values
      START_IL_TIMER();
      for(int j = 0 ; j < num_nonzero ; j++)
      {
            h_pij_vals[j] = h_pij_vals[ h_mapBfromA[j] ];
      }
	    END_IL_TIMER(_time_mapping);
           delete [] h_mapBfromA;
      delete [] h_Q;
      if (buffer_cpu) {free(buffer_cpu);}
      if (sol_handle) { checkCudaErrors(cusolverSpDestroy(sol_handle)); }


      
      
      std::cout << "Matrix B created" << std::endl;
      START_IL_TIMER();
           std::vector<int> v_row_ptr(h_pij_row_ptr, h_pij_row_ptr + (num_points+1));
      if (h_pij_row_ptr) { free(h_pij_row_ptr); }
      thrust::host_vector<int> row_temp(v_row_ptr);

           std::vector<int> v_col_ind(h_pij_col_ind, h_pij_col_ind + (num_nonzero));
      if (h_pij_col_ind) { free(h_pij_col_ind); }
      thrust::host_vector<int> col_temp(v_col_ind);
      
           std::vector<float> v_vals(h_pij_vals, h_pij_vals + (num_nonzero+1));
      if (h_pij_vals) {free(h_pij_vals);}
      thrust::host_vector<float> vals_temp(v_vals);
      //Update Pij vector to be passed to ComputeAttractiveForces
      
      
      pij_row_ptr_device = row_temp;
      pij_col_ind_device = col_temp;
      sparse_pij_device = vals_temp;
      END_IL_TIMER(_time_devicecopy);
      std::cout << "Completed permuting" << std::endl;
    }
   //Rabbit Order
   else if(opt.reorder==2){
    std::cout << "Entered Rabbit reorder branch..." << std::endl;
    START_IL_TIMER();
    int *h_pij_row_ptr = (int *)malloc((num_points+1)*sizeof(int));
      
      cudaMemcpy(h_pij_row_ptr, thrust::raw_pointer_cast(pij_row_ptr_device.data()), sizeof(int)*(num_points+1), cudaMemcpyDeviceToHost);

      int *h_pij_col_ind = (int *)malloc((num_nonzero)*sizeof(int));
      
      cudaMemcpy(h_pij_col_ind, thrust::raw_pointer_cast(pij_col_ind_device.data()), sizeof(int)*(num_nonzero), cudaMemcpyDeviceToHost);
    END_IL_TIMER(_time_hostcopy);

    START_IL_TIMER();
    std::vector<edge_list::edge> edges = gen_edgelist(h_pij_row_ptr, h_pij_col_ind, num_points, num_nonzero);
    //dump_edges("du", edges);
    
    //assert(0==1);
    auto adj = make_adj_list(num_points, edges);
    std::cerr << "Generating a permutation...\n";
    const double tstart = rabbit_order::now_sec();
    //-----------------------------------------------------
    const auto g = rabbit_order::aggregate(std::move(adj));
    const auto p = rabbit_order::compute_perm(g);
    //-----------------------------------------------------
    vint *perm1 = p.get();

    int *perm = (int *)malloc(sizeof(int)*num_points);

    //for(int i=0; i<num_points;i++){
    //  perm[i] = int(perm1[i]);
    //}
    //Convert vint array to int - ToDo: Just use ints in Rabbit Order
    for(int i = 0; i < num_points;i++)
    {
      perm[int(perm1[i])] = i;
    }
    std::cout << "Finished inverting the perm vector. Size: " << sizeof(perm1) <<std::endl;  
    free(perm1);
    
    END_IL_TIMER(_time_perm);

    assert(perm != NULL);
    //assert(sizeof(perm) == ((num_points)*sizeof(int)));
    
    int arr_size = sizeof(perm)/sizeof(perm[0]);
    std::cout << "Array Size: " << arr_size << std::endl;
    //save_cpu("permRab", perm, num_points );
    int *h_mapBfromA = NULL;
      
      cusolverSpHandle_t sol_handle = NULL;
      checkCudaErrors(cusolverSpCreate(&sol_handle));
       std::cout << "Created sparse solver handle" << std::endl;
      float *h_pij_vals = (float *)malloc((num_nonzero)*sizeof(float));
      
      cudaMemcpy(h_pij_vals, thrust::raw_pointer_cast(sparse_pij_device.data()), sizeof(float)*(num_nonzero), cudaMemcpyDeviceToHost);
      
      
     
      h_mapBfromA = (int *)malloc(sizeof(int)*num_nonzero);
      
      assert(NULL != h_mapBfromA);

       size_t size_perm = 0;
       void *buffer_cpu = NULL;
       
       START_IL_TIMER();
       checkCudaErrors(cusolverSpXcsrperm_bufferSizeHost(sol_handle,num_points ,num_points, num_nonzero, sparse_matrix_descriptor, h_pij_row_ptr,h_pij_col_ind, perm, perm, &size_perm));
       END_IL_TIMER(_time_reord_buff);

       buffer_cpu = (void*)malloc(sizeof(char)*size_perm);
       assert(NULL!=buffer_cpu);
       for(int j = 0; j< num_nonzero; j++) {
        h_mapBfromA[j] = j;
       }

       START_IL_TIMER();
       checkCudaErrors(cusolverSpXcsrpermHost(sol_handle, num_points, num_points, num_nonzero, sparse_matrix_descriptor, h_pij_row_ptr, h_pij_col_ind, perm, perm, h_mapBfromA, buffer_cpu) );
       END_IL_TIMER(_time_reorder);
      
      //Map the points
      float *h_pts = (float *)malloc(sizeof(float)*(num_points*2));
      float *h_pts_B = (float *)malloc(sizeof(float)*(num_points*2));

      cudaMemcpy(h_pts, thrust::raw_pointer_cast(points_device.data()), sizeof(float)*(num_points*2), cudaMemcpyDeviceToHost);


      START_IL_TIMER();
      for(int j = 0; j < num_points; j++) {
            h_pts_B[2*j] = h_pts[2*perm[j]];
            h_pts_B[2*j+1] = h_pts[2*perm[j]+1];      
      }
      
      std::vector<float> v_pts(h_pts_B, h_pts_B + (num_points*2));
      if (h_pij_vals) {free(h_pij_vals);}
      if (h_pts) {free(h_pts);}
      if (h_pts_B) {free(h_pts_B);}

      thrust::host_vector<float> pts_temp(v_pts);
      points_device = pts_temp;

      END_IL_TIMER(_time_map_points);
       //Map the values
      START_IL_TIMER();
      for(int j = 0 ; j < num_nonzero ; j++)
      {
            h_pij_vals[j] = h_pij_vals[ h_mapBfromA[j] ];
      }
      END_IL_TIMER(_time_mapping);
      delete [] h_mapBfromA;
      delete [] perm;
      if (buffer_cpu) {free(buffer_cpu);}
      if (sol_handle) { checkCudaErrors(cusolverSpDestroy(sol_handle)); }


      
      
      std::cout << "Matrix B created" << std::endl;
      START_IL_TIMER();
           std::vector<int> v_row_ptr(h_pij_row_ptr, h_pij_row_ptr + (num_points+1));
      if (h_pij_row_ptr) { free(h_pij_row_ptr); }
      thrust::host_vector<int> row_temp(v_row_ptr);

           std::vector<int> v_col_ind(h_pij_col_ind, h_pij_col_ind + (num_nonzero));
      if (h_pij_col_ind) { free(h_pij_col_ind); }
      thrust::host_vector<int> col_temp(v_col_ind);
      
           std::vector<float> v_vals(h_pij_vals, h_pij_vals + (num_nonzero+1));
      if (h_pij_vals) {free(h_pij_vals);}
      thrust::host_vector<float> vals_temp(v_vals);
      //Update Pij vector to be passed to ComputeAttractiveForces
      
      
      pij_row_ptr_device = row_temp;
      pij_col_ind_device = col_temp;
      sparse_pij_device = vals_temp;
      END_IL_TIMER(_time_devicecopy);
      std::cout << "Completed permuting" << std::endl;


   }
    //END_IL_REORDER(_time_tot_perm);
    std::cout << "Before nonzero2" << std::endl;
    if(opt.reorder != 9){ 
      std::cout << "before 2. csr2coo" << std::endl;
      tsnecuda::util::Csr2Coo(gpu_opt, coo_indices_device, pij_row_ptr_device,
                            pij_col_ind_device, num_points, num_nonzero);
    }
    std::cout << "Num nonzero 2: " << num_nonzero << std::endl;
    
    //tsnecuda::save_coo("coo_after_", coo_indices_device, num_nonzero);
    // FIT-TNSE Parameters
    int n_interpolation_points = 3;
    // float intervals_per_integer = 1;
    int min_num_intervals = 50;
    int N = num_points;
    // int D = 2;
    // The number of "charges" or s+2 sums i.e. number of kernel sums
    int n_terms = 4;
    int n_boxes_per_dim = min_num_intervals;

    // FFTW works faster on numbers that can be written as  2^a 3^b 5^c 7^d
    // 11^e 13^f, where e+f is either 0 or 1, and the other exponents are
    // arbitrary
    int allowed_n_boxes_per_dim[20] = {25,36, 50, 55, 60, 65, 70, 75, 80, 85, 90, 96, 100, 110, 120, 130, 140,150, 175, 200};
    if ( n_boxes_per_dim < allowed_n_boxes_per_dim[19] ) {
        //Round up to nearest grid point
        int chosen_i;
        for (chosen_i =0; allowed_n_boxes_per_dim[chosen_i]< n_boxes_per_dim; chosen_i++);
        n_boxes_per_dim = allowed_n_boxes_per_dim[chosen_i];
    }

    int n_total_boxes = n_boxes_per_dim * n_boxes_per_dim;
    int total_interpolation_points = n_total_boxes * n_interpolation_points * n_interpolation_points;
    int n_fft_coeffs_half = n_interpolation_points * n_boxes_per_dim;
    int n_fft_coeffs = 2 * n_interpolation_points * n_boxes_per_dim;
    int n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim;

    // FIT-TSNE Device Vectors
    thrust::device_vector<int> point_box_idx_device(N);
    thrust::device_vector<float> x_in_box_device(N);
    thrust::device_vector<float> y_in_box_device(N);
    thrust::device_vector<float> y_tilde_values(total_interpolation_points * n_terms);
    thrust::device_vector<float> x_interpolated_values_device(N * n_interpolation_points);
    thrust::device_vector<float> y_interpolated_values_device(N * n_interpolation_points);
    thrust::device_vector<float> potentialsQij_device(N * n_terms);
    thrust::device_vector<float> w_coefficients_device(total_interpolation_points * n_terms);
    thrust::device_vector<float> all_interpolated_values_device(
        n_terms * n_interpolation_points * n_interpolation_points * N);
    thrust::device_vector<float> output_values(
        n_terms * n_interpolation_points * n_interpolation_points * N);
    thrust::device_vector<int> all_interpolated_indices(
        n_terms * n_interpolation_points * n_interpolation_points * N);
    thrust::device_vector<int> output_indices(
        n_terms * n_interpolation_points * n_interpolation_points * N);
    thrust::device_vector<float> chargesQij_device(N * n_terms);
    thrust::device_vector<float> box_lower_bounds_device(2 * n_total_boxes);
    thrust::device_vector<float> box_upper_bounds_device(2 * n_total_boxes);
    thrust::device_vector<float> kernel_tilde_device(n_fft_coeffs * n_fft_coeffs);
    thrust::device_vector<thrust::complex<float>> fft_kernel_tilde_device(2 * n_interpolation_points_1d * 2 * n_interpolation_points_1d);
    thrust::device_vector<float> fft_input(n_terms * n_fft_coeffs * n_fft_coeffs);
    thrust::device_vector<thrust::complex<float>> fft_w_coefficients(n_terms * n_fft_coeffs * (n_fft_coeffs / 2 + 1));
    thrust::device_vector<float> fft_output(n_terms * n_fft_coeffs * n_fft_coeffs);

    // Easier to compute denominator on CPU, so we should just calculate y_tilde_spacing on CPU also
    float h = 1 / (float) n_interpolation_points;
    float y_tilde_spacings[n_interpolation_points];
    y_tilde_spacings[0] = h / 2;
    for (int i = 1; i < n_interpolation_points; i++) {
        y_tilde_spacings[i] = y_tilde_spacings[i - 1] + h;
    }
    float denominator[n_interpolation_points];
    for (int i = 0; i < n_interpolation_points; i++) {
        denominator[i] = 1;
        for (int j = 0; j < n_interpolation_points; j++) {
            if (i != j) {
                denominator[i] *= y_tilde_spacings[i] - y_tilde_spacings[j];
            }
        }
    }
    thrust::device_vector<float> y_tilde_spacings_device(y_tilde_spacings, y_tilde_spacings + n_interpolation_points);
    thrust::device_vector<float> denominator_device(denominator, denominator + n_interpolation_points);

    // Create the FFT Handles
    cufftHandle plan_kernel_tilde, plan_dft, plan_idft;;
    CufftSafeCall(cufftCreate(&plan_kernel_tilde));
    CufftSafeCall(cufftCreate(&plan_dft));
    CufftSafeCall(cufftCreate(&plan_idft));

    size_t work_size, work_size_dft, work_size_idft;
    int fft_dimensions[2] = {n_fft_coeffs, n_fft_coeffs};
    CufftSafeCall(cufftMakePlan2d(plan_kernel_tilde, fft_dimensions[0], fft_dimensions[1], CUFFT_R2C, &work_size));
    CufftSafeCall(cufftMakePlanMany(plan_dft, 2, fft_dimensions,
                                    NULL, 1, n_fft_coeffs * n_fft_coeffs,
                                    NULL, 1, n_fft_coeffs * (n_fft_coeffs / 2 + 1),
                                    CUFFT_R2C, n_terms, &work_size_dft));
    CufftSafeCall(cufftMakePlanMany(plan_idft, 2, fft_dimensions,
                                    NULL, 1, n_fft_coeffs * (n_fft_coeffs / 2 + 1),
                                    NULL, 1, n_fft_coeffs * n_fft_coeffs,
                                    CUFFT_C2R, n_terms, &work_size_idft));



    // Dump file
    float *host_ys = nullptr;
    std::ofstream dump_file;
    if (opt.get_dump_points()) {
        dump_file.open("pts_Y_" + std::to_string(opt.num_points));
        host_ys = new float[num_points * 2];
        dump_file << num_points << " " << 2 << std::endl;
    }

    #ifndef NO_ZMQ
        bool send_zmq = opt.get_use_interactive();
        zmq::context_t context(1);
        zmq::socket_t publisher(context, ZMQ_REQ);
        if (opt.get_use_interactive()) {

        // Try to connect to the socket
            if (opt.verbosity >= 1)
                std::cout << "Initializing Connection...." << std::endl;
            publisher.setsockopt(ZMQ_RCVTIMEO, opt.get_viz_timeout());
            publisher.setsockopt(ZMQ_SNDTIMEO, opt.get_viz_timeout());
            if (opt.verbosity >= 1)
                std::cout << "Waiting for connection to visualization for 10 secs...." << std::endl;
            publisher.connect(opt.get_viz_server());

            // Send the number of points we should be expecting to the server
            std::string message = std::to_string(opt.num_points);
            send_zmq = publisher.send(message.c_str(), message.length());

            // Wait for server reply
            zmq::message_t request;
            send_zmq = publisher.recv (&request);

            // If there's a time-out, don't bother.
            if (send_zmq) {
                if (opt.verbosity >= 1)
                    std::cout << "Visualization connected!" << std::endl;
            } else {
                std::cout << "No Visualization Terminal, continuing..." << std::endl;
                send_zmq = false;
            }
        }
    #else
        if (opt.get_use_interactive())
            std::cout << "This version is not built with ZMQ for interative viz. Rebuild with WITH_ZMQ=TRUE for viz." << std::endl;
    #endif

    if (opt.verbosity > 0) {
        std::cout << "done." << std::endl;
    }

    END_IL_TIMER(_time_init_fft);

    //create vector to record rep force computation time
    std::vector<float> rep_force_times;
    // Support for infinite iteration
    float time_mul, time_firstSPDM, time_secondSPDM, time_pijkern = 0.0;

    for (size_t step = 0; step != opt.iterations; step++) {

        START_IL_TIMER();
        float fill_value = 0;
        thrust::fill(w_coefficients_device.begin(), w_coefficients_device.end(), fill_value);
        thrust::fill(potentialsQij_device.begin(), potentialsQij_device.end(), fill_value);
        // Setup learning rate schedule
        if (step == opt.force_magnify_iters) {
            momentum = opt.post_exaggeration_momentum;
            attr_exaggeration = 1.0f;
        }
        END_IL_TIMER(_time_other);



        // Prepare the terms that we'll use to compute the sum i.e. the repulsive forces
        START_IL_TIMER();
        tsnecuda::ComputeChargesQij(chargesQij_device, points_device, num_points, n_terms);
        END_IL_TIMER(_time_compute_charges);

        // Compute Minimax elements
        START_IL_TIMER();
        auto minimax_iter = thrust::minmax_element(points_device.begin(), points_device.end());
        float min_coord = minimax_iter.first[0];
        float max_coord = minimax_iter.second[0];

        // Compute the number of boxes in a single dimension and the total number of boxes in 2d
        // auto n_boxes_per_dim = static_cast<int>(fmax(min_num_intervals, (max_coord - min_coord) / intervals_per_integer));

        tsnecuda::PrecomputeFFT2D(
            plan_kernel_tilde, max_coord, min_coord, max_coord, min_coord, n_boxes_per_dim, n_interpolation_points,
            box_lower_bounds_device, box_upper_bounds_device, kernel_tilde_device,
            fft_kernel_tilde_device);

        float box_width = ((max_coord - min_coord) / (float) n_boxes_per_dim);

        END_IL_TIMER(_time_precompute_2d);
        START_IL_TIMER();

        
        tsnecuda::NbodyFFT2D(
            plan_dft, plan_idft,
            N, n_terms, n_boxes_per_dim, n_interpolation_points,
            fft_kernel_tilde_device, n_total_boxes,
            total_interpolation_points, min_coord, box_width, n_fft_coeffs_half, n_fft_coeffs,
            fft_input, fft_w_coefficients, fft_output,
            point_box_idx_device, x_in_box_device, y_in_box_device, points_device,
            box_lower_bounds_device, y_tilde_spacings_device, denominator_device, y_tilde_values,
            all_interpolated_values_device, output_values, all_interpolated_indices,
            output_indices, w_coefficients_device, chargesQij_device, x_interpolated_values_device,
            y_interpolated_values_device, potentialsQij_device);

        END_IL_TIMER(_time_nbodyfft);

        PRINT_IL_TIMER(_time_nbodyfft);
        rep_force_times.push_back(((float) duration.count()) / 1000000.0);
        START_IL_TIMER();

        // Make the negative term, or F_rep in the equation 3 of the paper
        normalization = tsnecuda::ComputeRepulsiveForces(
            repulsive_forces_device, normalization_vec_device, points_device,
            potentialsQij_device, num_points, n_terms);

        END_IL_TIMER(_time_norm);
        START_IL_TIMER();


        // Calculate Attractive Forces            
        tsnecuda::ComputeAttractiveForces(gpu_opt,
                                              sparse_handle,
                                              sparse_matrix_descriptor,
                                              attractive_forces_device,
					                                    //pijqij,
                                              sparse_pij_device,
                                              //d_sp_pij_re,
                                              pij_row_ptr_device,
                                              pij_col_ind_device,
                                              coo_indices_device,
                                              //d_coo_re,
                                              points_device,
                                              ones_device,
                                              num_points,
                                              time_firstSPDM,
                                              time_secondSPDM,
                                              time_mul,
                                              time_pijkern,
                                              num_nonzero);

        END_IL_TIMER(_time_attr);
        START_IL_TIMER();

        // Apply Forces
        tsnecuda::ApplyForces(gpu_opt,
                                  points_device,
                                  attractive_forces_device,
                                  repulsive_forces_device,
                                  gains_device,
                                  old_forces_device,
                                  eta,
                                  normalization,
                                  momentum,
                                  attr_exaggeration,
                                  num_points,
                                  num_blocks);
        END_IL_TIMER(_time_apply_forces);
        // // Compute the gradient norm
        tsnecuda::util::SquareDeviceVector(attractive_forces_device, old_forces_device);
        thrust::transform(attractive_forces_device.begin(), attractive_forces_device.begin()+num_points,
                          attractive_forces_device.begin()+num_points, attractive_forces_device.begin(),
                          thrust::plus<float>());
        tsnecuda::util::SqrtDeviceVector(attractive_forces_device, attractive_forces_device);
        float grad_norm = thrust::reduce(
            attractive_forces_device.begin(), attractive_forces_device.begin() + num_points,
            0.0f, thrust::plus<float>()) / num_points;
        thrust::fill(attractive_forces_device.begin(), attractive_forces_device.end(), 0.0f);
        //END_IL_TIMER(_time_apply_forces);

        if (grad_norm < opt.min_gradient_norm) {
            if (opt.verbosity >= 1) std::cout << "Reached minimum gradient norm: " << grad_norm << std::endl;
            break;
        }

        if (opt.verbosity >= 1 && step % opt.print_interval == 0) {
            std::cout << "[Step " << step << "] Avg. Gradient Norm: " << grad_norm << std::endl;
        }

        

        #ifndef NO_ZMQ
            if (send_zmq) {
            zmq::message_t message(sizeof(float)*opt.num_points*2);
            thrust::copy(points_device.begin(), points_device.end(), static_cast<float*>(message.data()));
            bool res = false;
            res = publisher.send(message);
            zmq::message_t request;
            res = publisher.recv(&request);
            if (!res) {
                std::cout << "Server Disconnected, Not sending anymore for this session." << std::endl;
            }
            send_zmq = res;
            }
        #endif

        if (opt.get_dump_points() && step % opt.get_dump_interval() == 0) {
            thrust::copy(points_device.begin(), points_device.end(), host_ys);
            for (int i = 0; i < opt.num_points; i++) {
                dump_file << host_ys[i] << " " << host_ys[i + num_points] << std::endl;
            }
        }

        // // Handle snapshoting
        if (opt.return_style == tsnecuda::RETURN_STYLE::SNAPSHOT && step % snap_interval == 0 && opt.return_data != nullptr) {
          thrust::copy(points_device.begin(),
                       points_device.end(),
                       snap_num*opt.num_points*2 + opt.return_data);
          snap_num += 1;
        }

    }

    CufftSafeCall(cufftDestroy(plan_kernel_tilde));
    CufftSafeCall(cufftDestroy(plan_dft));
    CufftSafeCall(cufftDestroy(plan_idft));

    if (opt.verbosity > 0) {
        PRINT_IL_TIMER(_time_initialization);
        PRINT_IL_TIMER(_time_knn);
        PRINT_IL_TIMER(_time_knn2);
        PRINT_IL_TIMER(_time_normknn);
        PRINT_IL_TIMER(_time_symmetry);
        PRINT_IL_TIMER(_time_perm);
        PRINT_IL_TIMER(_time_reorder);
        PRINT_IL_TIMER(_time_reord_buff);
        PRINT_IL_TIMER(_time_mapping);
        PRINT_IL_TIMER(_time_hostcopy);
        PRINT_IL_TIMER(_time_devicecopy);
        PRINT_IL_TIMER(_time_tot_perm);
        PRINT_IL_TIMER(_time_init_low_dim);
        PRINT_IL_TIMER(_time_init_fft);
        PRINT_IL_TIMER(_time_compute_charges);
        PRINT_IL_TIMER(_time_precompute_2d);
        PRINT_IL_TIMER(_time_nbodyfft);
        PRINT_IL_TIMER(_time_norm);
        PRINT_IL_TIMER(_time_attr);
        PRINT_IL_TIMER(_time_apply_forces);
        PRINT_IL_TIMER(_time_other);
        PRINT_IL_TIMER(total_time);

        std::cout << "time_firstSPDM" << ": " << (time_firstSPDM) << "s" << std::endl;
        std::cout << "time_secondSPDM" << ": " << (time_secondSPDM) << "s" << std::endl;
        std::cout << "time_mul" << ": " << (time_mul) << "s" << std::endl;
        std::cout << "time_pijkern" << ": " << (time_pijkern ) << "s" << std::endl;


    }


    // Clean up the dump file if we are dumping points
    if (opt.get_dump_points()){
      delete[] host_ys;
      dump_file.close();
    }

    // Handle a once off return type
    if (opt.return_style == tsnecuda::RETURN_STYLE::ONCE && opt.return_data != nullptr) {
      thrust::copy(points_device.begin(), points_device.end(), opt.return_data);
    }

    // Handle snapshoting
    if (opt.return_style == tsnecuda::RETURN_STYLE::SNAPSHOT && opt.return_data != nullptr) {
      thrust::copy(points_device.begin(), points_device.end(), snap_num*opt.num_points*2 + opt.return_data);
    }
    if (opt.verbosity > 1) {
        std::ofstream reptimes_file;
        reptimes_file.open("./reptimes_" + std::to_string(opt.num_points/1000) + ".txt");
        //dump the values of sparse array Pij
        for (const auto &e : rep_force_times) reptimes_file << e << " ";
        //dump the indices of the values of Pij (COO format)
        //for (const auto &e : stl_pij_coo) pij_file << e << " ";
        //dump reordered values of sparse array pij
        //for(const auto &e : stl_reordered_pij) pij_file << e << " ";
        //dump the reordered indices of the values of Pij
        //for(const auto &e : stl_reordered_coo) pij_file << e << " ";

        reptimes_file.close();

    }
   
    // Return some final values
    opt.trained = true;
    opt.trained_norm = normalization;

    return;
}
