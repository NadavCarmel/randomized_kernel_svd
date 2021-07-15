Implementation of normalized Laplacian decomposition (for spectral clustering purposes etc.) by using a randomized SVD approach.

In fact, since we are usually interested at the eigen-vectors associated with the smallest eigen-values of the Laplacian, we *do not* approximate it at all. Rather, we compute the eigen-vectors associated with the *highest* eigen-values of the normalized *symmetric* kernel (adjacency) matrix.

The involves steps are:
- FPS sampling of the data 
- Kernel matrix low rank approximation (Nyström approximation)
- Normalize the approximated kernel
- Execution of randomized SVD over the normalized approximated kernel  

Input:
- X: data (np.array)

Output:
- U, S, Vh: the SVD outputs (np.arrays) 

Based on: 
- https://research.fb.com/blog/2014/09/fast-randomized-svd/
- Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions (https://arxiv.org/abs/0909.4061)
- Gil Shabbat's contribution.

