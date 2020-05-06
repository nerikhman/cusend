// note that this header file is special and does not use #pragma once

// CUSEND_ANNOTATION expands to __host__ __device__ when encountered by a
// CUDA-capable compiler

#if !defined(CUSEND_ANNOTATION)

#  ifdef __CUDACC__
#    define CUSEND_ANNOTATION __host__ __device__
#  else
#    define CUSEND_ANNOTATION
#  endif
#  define CUSEND_ANNOTATION_NEEDS_UNDEF

#elif defined(CUSEND_ANNOTATION_NEEDS_UNDEF)

#undef CUSEND_ANNOTATION
#undef CUSEND_ANNOTATION_NEEDS_UNDEF

#endif

