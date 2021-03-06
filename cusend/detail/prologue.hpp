// note that this header file is special and does not use #pragma once

#ifndef CUSEND_INCLUDE_LEVEL

// the first time this header is #included, this branch is processed

// this definition communicates that the stack is empty
// and that these macros should be undefined by epilogue.hpp
#define CUSEND_INCLUDE_LEVEL 0

// include preprocessor headers
#include "preprocessor.hpp"

#else

// any other time this header is #included, this branch is processed

// this push to the stack communicates with epilogue.hpp
// that these macros are not ready to be undefined.
#pragma push_macro("CUSEND_INCLUDE_LEVEL")
#undef CUSEND_INCLUDE_LEVEL
#define CUSEND_INCLUDE_LEVEL 1

#endif

