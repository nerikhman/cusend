// note that this header file is special and does not use #pragma once

// CUSEND_HAS_EXCEPTIONS indicates whether or not exception support is available.

#ifndef CUSEND_HAS_EXCEPTIONS

#  if defined(__CUDACC__)
#    if !defined(__CUDA_ARCH__)
#      define CUSEND_HAS_EXCEPTIONS __cpp_exceptions
#    else
#      define CUSEND_HAS_EXCEPTIONS 0
#    endif
#  else
#    define CUSEND_HAS_EXCEPTIONS __cpp_exceptions
#  endif

#elif defined(CUSEND_HAS_EXCEPTIONS)
#  undef CUSEND_HAS_EXCEPTIONS
#endif

