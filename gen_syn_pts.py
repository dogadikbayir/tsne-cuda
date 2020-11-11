import numpy as np
import sys
import os
import gensim
from gensim.models import KeyedVectors as Word2VecLoader
import matplotlib.pyplot as plt
import matplotlib as mpl

#import faiss
import time
from tsnecuda import TSNE

from sklearn.datasets import make_classification

def fvecs_read(filename, c_contiguous=True):
  fv = np.fromfile(filename, dtype=np.float32)
  if fv.size == 0:
    return np.zeros((0,0))

  dim = fv.view(np.int32)[0]
  assert dim > 0

  fv = fv.reshape(-1, 1 + dim)
  if not all(fv.view(np.int32)[:,0] == dim):
    raise IOError("Non-uniform vector sizes in " + filename)

  fv = fv[:, 1:]
  if c_contiguous:
    fv = fv.copy()
  return fv

def bvecs_read(filename, c_contiguous=True):
  fv = np.fromfile(filename, dtype=np.uint8)
  if fv.size == 0:
    return np.zeros((0,0))

  dim = fv.view(np.uint8)[0]
  assert dim > 0

  fv = fv.reshape(-1, 4 + dim)
  if not all(fv.view(np.uint8)[:,0] == dim):
    raise IOError("Non-uniform vector sizes in " + filename)

  fv = fv[:, 4:]
  if c_contiguous:
    fv = fv.copy()
  fv = fv.astype('float32')
  return fv


num_points = int(sys.argv[1])
num_dims = int(sys.argv[2])
sep = int(sys.argv[3])
k = int(sys.argv[4])

option = int(sys.argv[5])
reorder = int(sys.argv[6])
num_clusters = int(sys.argv[7])
iterations = int(sys.argv[8])
d_intervals = int(sys.argv[9])
kNumCellsToProbe = int(sys.argv[10])
side = int(sys.argv[11])

step_freq = int(sys.argv[12])
log = int(sys.argv[13])
avg_freq = int(sys.argv[14])
#lr = int(sys.argv[11])
d_pts = False
if d_intervals > 0:
  d_pts = True


if option == 0:
  np.savetxt(str(num_points) + ".data", x, delimiter=" ", fmt='%f')

elif option == 1:
  x = np.loadtxt('/mnt/home/dikbayir/datasets/mnist.csv', delimiter=",")
  X_emb = TSNE(avg_freq=avg_freq, log=log, step_freq=step_freq, side=100,reorder=reorder, n_iter=iterations,verbose=True, dump_points=d_pts, dump_interval=d_intervals).fit_transform(x)
  
  mpl.rcParams['agg.path.chunksize'] = 10000
  plt.figure(figsize=(10,10), dpi=600)
  plt.scatter(X_emb[:,0], X_emb[:,1], marker=',', s=72./1200)
  plt.axis('off')
  plt.savefig('res_mnist_' + str(iterations) + '.png')
  
  #plt.figure(figsize=(10,10), dpi=600)
  #plt.plot(X_emb[:,0], X_emb[:,1], markersize=, marker=',')
  
  #plt.savefig('mnist.png')
elif option == 2:
  x, y = make_classification(n_samples=num_points, n_features=num_dims, n_redundant=int(num_dims/2), n_informative=int(num_dims/2), class_sep=sep, n_clusters_per_class=1, scale=10.0, n_classes=num_clusters,shuffle=True,random_state=42)

  x = x.astype('float32')

  X_emb = TSNE(avg_freq=avg_freq, log=log, step_freq=step_freq, side=side, reorder=reorder,n_iter=iterations,verbose=True,
      dump_points=d_pts, dump_interval=d_intervals
      ).fit_transform(x)

elif option == 3:
  x = np.loadtxt('/mnt/home/dikbayir/datasets/vectors2.txt', delimiter = ' ')
  x = x[0:num_points]
  X_emb = TSNE(avg_freq=avg_freq, log=log, step_freq=step_freq, side=side, reorder=reorder, n_iter=iterations, verbose=True,
      dump_points=d_pts, dump_interval=d_intervals
      ).fit_transform(x)
elif option == 4:
  word2vec_model = '/mnt/home/dikbayir/datasets/GN.bin'
  model = Word2VecLoader.load_word2vec_format(word2vec_model, binary=word2vec_model.endswith('bin'))
  x = model.vectors
  x=x[:num_points,:]
  X_emb = TSNE(avg_freq=avg_freq, log=log, step_freq=step_freq, side=side, reorder=reorder,n_iter=iterations, verbose=True,
      dump_points=d_pts, dump_interval=d_intervals
      ).fit_transform(x)

elif option == 5:
  x = fvecs_read('/mnt/home/dikbayir/datasets/deep10M.fvecs')
  x = x[:num_points,:]
  print(x.shape)
  
  X_emb = TSNE(avg_freq=avg_freq, log=log, step_freq=step_freq, side=side, reorder=reorder, n_iter=iterations, verbose=True,
      dump_points=d_pts, dump_interval=d_intervals).fit_transform(x)
elif option == 6:
   x = bvecs_read('/mnt/home/dikbayir/datasets/100mSIFT.bvecs')
   x = x[:num_points,:]
   
   X_emb = TSNE(avg_freq=avg_freq, log=log, step_freq=step_freq, side=side, reorder=reorder, n_iter=iterations, verbose=True, dump_points=d_pts, dump_interval=d_intervals).fit_transform(x)

else:
  print("Inside faiss branch")
  start = time.perf_counter()
  res = faiss.StandardGpuResources()

  index_flat = faiss.index_factory(num_dims, "IVF4096,PQ64")
  #gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
  co = faiss.GpuClonerOptions()
  
  gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat, co)

  #index.train(x)
  print("train")
  start_train = time.perf_counter()
  gpu_index_flat.train(x)
  print("Train time: " + str(time.perf_counter()-start_train) + "s")
  
  print("add index")
  start_add = time.perf_counter()
  gpu_index_flat.add(x)
  print("Add time: " + str(time.perf_counter()-start_add) + "s")

  #gpu_index_flat.search(x[:5], 123)

  
  #nlist=20
  
    #quantizer = faiss.IndexFlatL2(num_dims)
  #index = faiss.IndexIVFFlat(quantizer, num_dims, nlist)
  
  #index.nprobe = 20

  #index.train(x)
  #index.add(x)

  #print(index.is_trained)
  #print(index.ntotal)
  print("Starting search:")
  start_search = time.perf_counter()
  D, I = gpu_index_flat.search(x, k)
  print("Search time: " + str(time.perf_counter()-start_search) + "s")

  print (type(I))
  #print (D)

  print("Total Elapsed time: " + str(time.perf_counter()-start) + "s")

  #print (I)
  np.savetxt("faiss_res.txt", I, delimiter=" ")


