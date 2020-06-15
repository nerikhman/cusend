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
#include "detail/default_just.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Type, class... Types>
using just_member_function_t = decltype(std::declval<Type>().just(std::declval<Types>()...));

template<class Type, class... Types>
using has_just_member_function = is_detected<just_member_function_t, Type, Types...>;


template<class Type, class... Types>
using just_free_function_t = decltype(just(std::declval<Type>(), std::declval<Types>()...));

template<class Type, class... Types>
using has_just_free_function = is_detected<just_free_function_t, Type, Types...>;


// this is the type of the just CPO
struct dispatch_just
{
  CUSEND_EXEC_CHECK_DISABLE
  template<class Type, class... Types,
           CUSEND_REQUIRES(has_just_member_function<Type&&,Types&&...>::value)
          >
  CUSEND_ANNOTATION
  constexpr just_member_function_t<Type&&,Types&&...>
    operator()(Type&& value, Types&&... values) const
  {
    return std::forward<Type>(value).just(std::forward<Types>(values)...);
  }

  CUSEND_EXEC_CHECK_DISABLE
  template<class Type, class... Types,
           CUSEND_REQUIRES(!has_just_member_function<Type&&,Types&&...>::value),
           CUSEND_REQUIRES(has_just_free_function<Type&&,Types&&...>::value)
          >
  CUSEND_ANNOTATION
  constexpr just_free_function_t<Type&&,Types&&...>
    operator()(Type&& value, Types&&... values) const
  {
    return just(std::forward<Type>(value), std::forward<Types>(values)...);
  }

  CUSEND_EXEC_CHECK_DISABLE
  template<class Type, class... Types,
           CUSEND_REQUIRES(!has_just_member_function<Type&&,Types&&...>::value),
           CUSEND_REQUIRES(!has_just_free_function<Type&&,Types&&...>::value),
           CUSEND_REQUIRES(is_detected<default_just_t,Type&&,Types&&...>::value)
          >
  CUSEND_ANNOTATION
  constexpr default_just_t<Type&&,Types&&...>
    operator()(Type&& value, Types&&... values) const
  {
    return detail::default_just(std::forward<Type>(value), std::forward<Types>(values)...);
  }


  CUSEND_EXEC_CHECK_DISABLE
  template<class... Types,
           CUSEND_REQUIRES(is_detected<default_just_t,Types&&...>::value)
          >
  CUSEND_ANNOTATION
  constexpr default_just_t<Types&&...>
    operator()(Types&&... values) const
  {
    return detail::default_just(std::forward<Types>(values)...);
  }
};


} // end detail


namespace
{


// define the just customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& just = detail::static_const<detail::dispatch_just>::value;
#else
const __device__ detail::dispatch_just just;
#endif


} // end anonymous namespace


template<class... Types>
using just_t = decltype(just(std::declval<Types>()...));


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

