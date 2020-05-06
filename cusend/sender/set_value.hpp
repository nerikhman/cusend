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
#include "../detail/static_const.hpp"
#include "../detail/type_traits/is_detected.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class R, class... Args>
using set_value_member_function_t = decltype(std::declval<R>().set_value(std::declval<Args>()...));

template<class R, class... Args>
using has_set_value_member_function = is_detected<set_value_member_function_t, R, Args...>;


template<class R, class... Args>
using set_value_free_function_t = decltype(set_value(std::declval<R>(), std::declval<Args>()...));

template<class R, class... Args>
using has_set_value_free_function = is_detected<set_value_free_function_t, R, Args...>;


// this is the type of set_value
struct dispatch_set_value
{
  CUSEND_EXEC_CHECK_DISABLE
  template<class R, class... Args,
           CUSEND_REQUIRES(has_set_value_member_function<R&&,Args&&...>::value)
          >
  CUSEND_ANNOTATION
  constexpr auto operator()(R&& r, Args&&... args) const ->
    decltype(std::forward<R>(r).set_value(std::forward<Args>(args)...))
  {
    return std::forward<R>(r).set_value(std::forward<Args>(args)...);
  }

  CUSEND_EXEC_CHECK_DISABLE
  template<class R, class... Args,
           CUSEND_REQUIRES(!has_set_value_member_function<R&&,Args&&...>::value and
                          has_set_value_free_function<R&&,Args&&...>::value)
           >
  CUSEND_ANNOTATION
  constexpr auto operator()(R&& r, Args&&... args) const ->
    decltype(set_value(std::forward<R>(r), std::forward<Args>(args)...))
  {
    return set_value(std::forward<R>(r), std::forward<Args>(args)...);
  }
};


} // end detail



namespace
{


// define the set_value customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& set_value = detail::static_const<detail::dispatch_set_value>::value;
#else
const __device__ detail::dispatch_set_value set_value;
#endif


} // end anonymous namespace


template<class T, class... Args>
using set_value_t = decltype(CUSEND_NAMESPACE::set_value(std::declval<T>(), std::declval<Args>()...));


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

