// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "../detail/prologue.hpp"

#include <utility>
#include "../detail/type_traits/is_detected.hpp"
#include "../detail/static_const.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
using start_member_function_t = decltype(std::declval<T>().start());

template<class T>
using has_start_member_function = is_detected<start_member_function_t, T>;


template<class T>
using start_free_function_t = decltype(start(std::declval<T>()));

template<class T>
using has_start_free_function = is_detected<start_free_function_t, T>;


// this is the type of start
struct dispatch_start
{
  CUDEX_EXEC_CHECK_DISABLE
  template<class T, 
           CUDEX_REQUIRES(has_start_member_function<T&&>::value)
          >
  CUDEX_ANNOTATION
  constexpr auto operator()(T&& arg) const ->
    decltype(std::forward<T>(arg).start())
  {
    return std::forward<T>(arg).start();
  }

  CUDEX_EXEC_CHECK_DISABLE
  template<class T,
           CUDEX_REQUIRES(!has_start_member_function<T&&>::value and
                          has_start_free_function<T&&>::value)
          >
  CUDEX_ANNOTATION
  constexpr auto operator()(T&& arg) const ->
    decltype(start(std::forward<T>(arg)))
  {
    return start(std::forward<T>(arg));
  }
};


} // end detail


namespace
{


// define the start customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& start = detail::static_const<detail::dispatch_start>::value;
#else
const __device__ detail::dispatch_start start;
#endif


} // end anonymous namespace


template<class T>
using start_t = decltype(CUDEX_NAMESPACE::start(std::declval<T>()));


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

