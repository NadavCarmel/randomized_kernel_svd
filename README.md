Implementation of normalized Laplacian decomposition (for spectral clustering purposes etc.) by using a randomized SVD approach.
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

