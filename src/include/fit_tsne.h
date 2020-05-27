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

#ifndef NO_ZMQ
    #include <zmq.hpp>
#endif

namespace tsnecuda {
std::vector<std::string> split(std::string str, std::string del );
void RunTsne(tsnecuda::Options &opt, tsnecuda::GpuOptions &gpu_opt);
}


#endif
