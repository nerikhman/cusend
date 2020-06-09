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

#include "../prologue.hpp"

#include <utility>
#include "../static_const.hpp"
#include "../type_traits/is_detected.hpp"
#include "default_transform.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class S, class F>
using transform_member_function_t = decltype(std::declval<S>().transform(std::declval<F>()));

template<class S, class F>
using has_transform_member_function = is_detected<transform_member_function_t, S, F>;


template<class S, class F>
using transform_free_function_t = decltype(transform(std::declval<S>(), std::declval<F>()));

template<class S, class F>
using has_transform_free_function = is_detected<transform_free_function_t, S, F>;


// this is the type of the transform CPO
struct dispatch_transform
{
  CUSEND_EXEC_CHECK_DISABLE
  template<class Sender, class Function,
           CUSEND_REQUIRES(has_transform_member_function<Sender&&,Function&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr transform_member_function_t<Sender&&,Function&&>
    operator()(Sender&& predecessor, Function&& continuation) const
  {
    return std::forward<Sender>(predecessor).transform(std::forward<Function>(continuation));
  }

  CUSEND_EXEC_CHECK_DISABLE
  template<class Sender, class Function,
           CUSEND_REQUIRES(!has_transform_member_function<Sender&&,Function&&>::value),
           CUSEND_REQUIRES(has_transform_free_function<Sender&&,Function&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr transform_free_function_t<Sender&&,Function&&>
    operator()(Sender&& predecessor, Function&& continuation) const
  {
    return transform(std::forward<Sender>(predecessor), std::forward<Function>(continuation));
  }

  template<class Sender, class Function,
           CUSEND_REQUIRES(!has_transform_member_function<Sender&&,Function&&>::value),
           CUSEND_REQUIRES(!has_transform_free_function<Sender&&,Function&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr default_transform_t<Sender&&,Function&&>
    operator()(Sender&& predecessor, Function&& continuation) const
  {
    return detail::default_transform(std::forward<Sender>(predecessor), std::forward<Function>(continuation));
  }
};


namespace
{


// define the transform customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& transform = detail::static_const<detail::dispatch_transform>::value;
#else
const __device__ detail::dispatch_transform transform;
#endif


} // end anonymous namespace


template<class S, class F>
using transform_t = decltype(detail::transform(std::declval<S>(), std::declval<F>()));


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../epilogue.hpp"

