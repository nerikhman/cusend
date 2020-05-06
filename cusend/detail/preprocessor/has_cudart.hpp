// note that this header file is special and does not use #pragma once

// CUSEND_HAS_CUDART indicates whether or not the CUDA Runtime API is available.

#ifndef CUSEND_HAS_CUDART

#  if defined(__CUDACC__)
#    if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__>= 350 && defined(__CUDACC_RDC__))
#      define CUSEND_HAS_CUDART 1
#    else
#      define CUSEND_HAS_CUDART 0
#    endif
#  else
#    define CUSEND_HAS_CUDART __has_include(<cuda_runtime_api.h>)
#  endif

#elif defined(CUSEND_HAS_CUDART)
#  undef CUSEND_HAS_CUDART
#endif

