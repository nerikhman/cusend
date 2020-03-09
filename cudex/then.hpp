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

#include "detail/prologue.hpp"

#include <utility>
#include "detail/default_then.hpp"
#include "detail/static_const.hpp"
#include "detail/type_traits/has_then.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


// this is the type of then
struct then_customization_point
{
  CUDEX_EXEC_CHECK_DISABLE
  template<class S, class F,
           CUDEX_REQUIRES(has_then_member_function<S&&,F&&>::value)
          >
  CUDEX_ANNOTATION
  constexpr auto operator()(S&& s, F&& f) const ->
    decltype(std::forward<S>(s).then(std::forward<F>(f)))
  {
    return std::forward<S>(s).then(std::forward<F>(f));
  }


  CUDEX_EXEC_CHECK_DISABLE
  template<class S, class F,
           CUDEX_REQUIRES(!has_then_member_function<S&&,F&&>::value),
           CUDEX_REQUIRES(has_then_free_function<S&&,F&&>::value)
          >
  CUDEX_ANNOTATION
  constexpr auto operator()(S&& s, F&& f) const ->
    decltype(then(std::forward<S>(s), std::forward<F>(f)))
  {
    return then(std::forward<S>(s), std::forward<F>(f));
  }


  CUDEX_EXEC_CHECK_DISABLE
  template<class S, class F,
           CUDEX_REQUIRES(!has_then_member_function<S&&,F&&>::value),
           CUDEX_REQUIRES(!has_then_free_function<S&&,F&&>::value)
          >
  CUDEX_ANNOTATION
  constexpr auto operator()(S&& s, F&& f) const ->
    decltype(detail::default_then(std::forward<S>(s), std::forward<F>(f)))
  {
    return detail::default_then(std::forward<S>(s), std::forward<F>(f));
  }
};


} // end detail


namespace
{


// define the then customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& then = detail::static_const<detail::then_customization_point>::value;
#else
const __device__ detail::then_customization_point then;
#endif


} // end anonymous namespace


CUDEX_NAMESPACE_CLOSE_BRACE


#include "detail/epilogue.hpp"

