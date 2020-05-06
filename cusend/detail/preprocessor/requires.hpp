// note that this header file is special and does not use #pragma once

// The CUSEND_REQUIRES() macro may be used in a function template's parameter list
// to simulate Concepts.
//
// For example, to selectively enable a function template only for integer types,
// we could do something like this:
//
//     template<class Integer,
//              CUSEND_REQUIRES(std::is_integral<Integer>::value)
//             >
//     Integer plus_one(Integer x)
//     {
//       return x + 1;
//     }
//

#ifndef CUSEND_REQUIRES

#  define CUSEND_CONCATENATE_IMPL(x, y) x##y

#  define CUSEND_CONCATENATE(x, y) CUSEND_CONCATENATE_IMPL(x, y)

#  define CUSEND_MAKE_UNIQUE(x) CUSEND_CONCATENATE(x, __COUNTER__)

#  define CUSEND_REQUIRES_IMPL(unique_name, ...) bool unique_name = true, typename std::enable_if<(unique_name and __VA_ARGS__)>::type* = nullptr

#  define CUSEND_REQUIRES(...) CUSEND_REQUIRES_IMPL(CUSEND_MAKE_UNIQUE(__deduced_true), __VA_ARGS__)

#elif defined(CUSEND_REQUIRES)

#  ifdef CUSEND_CONCATENATE_IMPL
#    undef CUSEND_CONCATENATE_IMPL
#  endif

#  ifdef CUSEND_CONCATENATE
#    undef CUSEND_CONCATENATE
#  endif

#  ifdef CUSEND_MAKE_UNIQUE
#    undef CUSEND_MAKE_UNIQUE
#  endif

#  ifdef CUSEND_REQUIRES_IMPL
#    undef CUSEND_REQUIRES_IMPL
#  endif

#  ifdef CUSEND_REQUIRES
#    undef CUSEND_REQUIRES
#  endif

#endif

