// note that this header file is special and does not use #pragma once

#if !defined(CUSEND_NAMESPACE)

// this branch is taken the first time this header is included

#  if defined(CUSEND_NAMESPACE_OPEN_BRACE) or defined(CUSEND_NAMESPACE_CLOSE_BRACE)
#    error "Either all of CUSEND_NAMESPACE, CUSEND_NAMESPACE_OPEN_BRACE, and CUSEND_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

#  define CUSEND_NAMESPACE cusend
#  define CUSEND_NAMESPACE_OPEN_BRACE namespace cusend {
#  define CUSEND_NAMESPACE_CLOSE_BRACE }
#  define CUSEND_NAMESPACE_NEEDS_UNDEF

#elif defined(CUSEND_NAMESPACE_NEEDS_UNDEF)

// this branch is taken the second time this header is included

#  undef CUSEND_NAMESPACE
#  undef CUSEND_NAMESPACE_OPEN_BRACE
#  undef CUSEND_NAMESPACE_CLOSE_BRACE
#  undef CUSEND_NAMESPACE_NEEDS_UNDEF

#elif defined(CUSEND_NAMESPACE) or defined(CUSEND_NAMESPACE_OPEN_BRACE) or defined(CUSEND_CLOSE_BRACE)

// this branch is taken the first time this header is included, and the user has misconfigured these namespace-related symbols

#  if !defined(CUSEND_NAMESPACE) or !defined(CUSEND_NAMESPACE_OPEN_BRACE) or !defined(CUSEND_NAMESPACE_CLOSE_BRACE)
#    error "Either all of CUSEND_NAMESPACE, CUSEND_NAMESPACE_OPEN_BRACE, and CUSEND_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

#endif

