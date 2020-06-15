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

#include <type_traits>
#include "../detail/type_traits/conjunction.hpp"
#include "../detail/type_traits/is_detected.hpp"
#include "connect.hpp"
#include "is_sender_to.hpp"
#include "is_typed_sender.hpp"
#include "set_done.hpp"
#include "set_error.hpp"
#include "set_value.hpp"
#include "sender_traits.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{
namespace is_typed_sender_to_detail
{


template<class...>
struct type_list {};

template<class R, class TypeList>
struct can_set_value_with;

template<class R, class... Types>
struct can_set_value_with<R, type_list<Types...>> : is_detected<set_value_t, R, Types...> {};



template<class R, class ListOfTypeLists>
struct can_set_value_with_each;


template<class R, class... TypeLists>
struct can_set_value_with_each<R, type_list<TypeLists...>> : conjunction<can_set_value_with<R,TypeLists>...> {};


template<class S, class R>
struct can_set_value
{
  template<class S_,
           class ValueTypes = typename sender_traits<S_>::template value_types<type_list, type_list>
          >
  static constexpr bool test(int)
  {
    return can_set_value_with_each<R, ValueTypes>::value;
  }

  template<class>
  static constexpr bool test(...)
  {
    return false;
  }

  constexpr static bool value = test<S>(0);
};


template<class R, class TypeList>
struct can_set_error_with_each;

template<class R, class... Types>
struct can_set_error_with_each<R, type_list<Types...>> : conjunction<
  is_detected<set_error_t, R, Types>...
>
{};


template<class S, class R>
struct can_set_error
{
  template<class S_,
           class ErrorTypes = typename sender_traits<S_>::template error_types<type_list>
          >
  static constexpr bool test(int)
  {
    return can_set_error_with_each<R, ErrorTypes>::value;
  }

  template<class>
  static constexpr bool test(...)
  {
    return false;
  }

  constexpr static bool value = test<S>(0);
};


template<class S, class R>
struct can_set_done_if_need_be
{
  template<class S_,
           bool sends_done = sender_traits<S_>::sends_done,
           CUSEND_REQUIRES(is_detected<set_done_t, R>::value)
          >
  static constexpr bool test(int)
  {
    return true;
  }

  template<class>
  static constexpr bool test(...)
  {
    return false;
  }

  static constexpr bool value = test<S>(0);
};


} // end is_typed_sender_to_detail
} // end detail


template<class S, class ConnectReceiver, class ValueReceiver = ConnectReceiver, class ErrorReceiver = ValueReceiver, class DoneReceiver = ErrorReceiver>
using is_typed_sender_to = detail::conjunction<
  is_typed_sender<S>,
  detail::is_detected<connect_t,S,ConnectReceiver>,
  detail::is_typed_sender_to_detail::can_set_value<S,ValueReceiver>,
  detail::is_typed_sender_to_detail::can_set_error<S,ErrorReceiver>,
  detail::is_typed_sender_to_detail::can_set_done_if_need_be<S,DoneReceiver>
>;


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

