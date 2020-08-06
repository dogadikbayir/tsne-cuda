/**
 * @brief Barnes-Hut T-SNE implementation O(Nlog(N))
  *
 * @file bh_tsne.h
 * @author David Chan
 * @date 2018-04-15
 */

#ifndef BH_TSNE_H
#define BH_TSNE_H

#include "common.h"
#include "include/options.h"
#include "include/util/cuda_utils.h"
#include "include/util/math_utils.h"
#include "include/util/matrix_broadcast_utils.h"
#include "include/util/reduce_utils.h"
#include "include/util/distance_utils.h"
#include "include/util/random_utils.h"
#include "include/util/thrust_utils.h"
#include "include/util/thrust_transform_functions.h"

#include "include/kernels/apply_forces.h"
#include "include/kernels/attr_forces.h"
#include "include/kernels/perplexity_search.h"
#include "include/kernels/nbodyfft.h"
#include "include/kernels/rep_forces.h"

#include "rabbit_order.hpp"
#include "edge_list.hpp"
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/count.hpp>

#ifndef NO_ZMQ
    #include <zmq.hpp>
#endif

namespace tsnecuda {

using rabbit_order::vint;

typedef std::vector<std::vector<std::pair<vint, float>>> adjacency_list;
///typedef std::tuple<vint, vint, float> edge;


std::vector<std::string> split(std::string str, std::string del );
std::vector<edge_list::edge> gen_edgelist(int *row_ptr, int *col_ind, int num_points, int num_nnz);

template <typename T>
void tsnecuda::save_coo(std::string filename, thrust::device_vector<T> device_vec, int size_coo );

template<typename RandomAccessRange>
adjacency_list make_adj_list(const vint n, const RandomAccessRange& es);

template <typename T>
void save_coo(std::string filename, thrust::device_vector<T> device_vec, int num_nonzero); 

void RunTsne(tsnecuda::Options &opt, tsnecuda::GpuOptions &gpu_opt);
}


#endif
