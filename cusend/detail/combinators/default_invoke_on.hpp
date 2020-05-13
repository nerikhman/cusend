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

#include <exception>
#include <type_traits>
#include <utility>
#include "../../execution/executor/is_executor.hpp"
#include "../../execution/executor/is_executor_of.hpp"
#include "../../sender/is_receiver_of.hpp"
#include "../../sender/sender_base.hpp"
#include "../execute_operation.hpp"
#include "../functional/closure.hpp"
#include "../functional/compose.hpp"
#include "../receiver_as_invocable.hpp"
#include "../type_traits/is_invocable.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<template<class...> class Tuple, class Invocable, class... Args>
using tuple_of_invoke_result_or_empty_tuple_t = typename std::conditional<
  // check for a void result
  std::is_void<invoke_result_t<Invocable,Args...>>::value,
  // avoid instantiating Tuple with a void parameter
  Tuple<>,
  // instantiate Tuple as normal
  Tuple<invoke_result_t<Invocable,Args...>>
>::type;


// this is a sender that invokes a function on an executor and sends the result to a receiver
template<class Executor, class Invocable>
class invoke_sender
{
  public:
    template<template<class...> class Variant, template<class...> class Tuple>
    using value_types = Variant<tuple_of_invoke_result_or_empty_tuple_t<Tuple, Invocable>>;

    template<template<class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    constexpr static bool sends_done = true;

    template<class OtherInvocable,
             CUSEND_REQUIRES(std::is_constructible<Invocable,OtherInvocable&&>::value)
            >
    CUSEND_ANNOTATION
    invoke_sender(const Executor& ex, OtherInvocable&& invocable)
      : ex_(ex), invocable_(std::forward<OtherInvocable>(invocable))
    {}

    invoke_sender(const invoke_sender&) = default;

    invoke_sender(invoke_sender&&) = default;

    // the type of operation returned by connect
    template<class Receiver>
    using operation = execute_operation<
      Executor,
      function_composition<
        receiver_as_invocable<Receiver>,
        Invocable
      >
    >;


    template<class Receiver,
             CUSEND_REQUIRES(is_receiver_of<Receiver, invoke_result_t<Invocable>>::value)
            >
    CUSEND_ANNOTATION
    operation<Receiver&&> connect(Receiver&& r) &&
    {
      auto composition = detail::compose(detail::as_invocable(std::forward<Receiver>(r)), std::move(invocable_));
      return detail::make_execute_operation(ex_, std::move(composition));
    }


    // this overload allows makes invoke_sender a "multi-shot" sender when Invocable is copyable
    template<class Receiver,
             CUSEND_REQUIRES(std::is_copy_constructible<Invocable>::value),
             CUSEND_REQUIRES(is_receiver_of<Receiver, invoke_result_t<Invocable>>::value)
            >
    CUSEND_ANNOTATION
    operation<Receiver&&> connect(Receiver&& r) const &
    {
      auto composition = detail::compose(detail::as_invocable(std::forward<Receiver>(r)), invocable_);
      return detail::make_execute_operation(ex_, std::move(composition));
    }

    
    template<class OtherExecutor,
             CUSEND_REQUIRES(execution::is_executor<OtherExecutor>::value)
            >
    CUSEND_ANNOTATION
    invoke_sender<OtherExecutor, Invocable> on(const OtherExecutor& ex) &&
    {
      return {ex, std::move(invocable_)};
    }


    template<class OtherExecutor,
             CUSEND_REQUIRES(execution::is_executor<OtherExecutor>::value),
             CUSEND_REQUIRES(std::is_copy_constructible<Invocable>::value)
            >
    CUSEND_ANNOTATION
    invoke_sender<OtherExecutor, Invocable> on(const OtherExecutor& ex) const &
    {
      return {ex, invocable_};
    }

  private:
    Executor ex_;
    Invocable invocable_;
};


template<class Executor, class Invocable,
         CUSEND_REQUIRES(execution::is_executor_of<Executor,Invocable>::value)
        >
CUSEND_ANNOTATION
invoke_sender<Executor, decay_t<Invocable>>
  default_invoke_on(const Executor& ex, Invocable&& f)
{
  return {ex, std::forward<Invocable>(f)};
}


template<class Executor, class Invocable,
         class Arg1, class... Args,
         CUSEND_REQUIRES(execution::is_executor<Executor>::value),
         CUSEND_REQUIRES(detail::is_invocable<Invocable,Arg1,Args...>::value)
        >
CUSEND_ANNOTATION
invoke_sender<Executor, closure<decay_t<Invocable>, decay_t<Arg1>, decay_t<Args>...>>
  default_invoke_on(const Executor& ex, Invocable&& f, Arg1&& arg1, Args&&... args)
{
  return detail::default_invoke_on(ex, detail::bind(std::forward<Invocable>(f), std::forward<Arg1>(arg1), std::forward<Args>(args)...));
}


template<class Executor, class Invocable, class... Args>
using default_invoke_on_t = decltype(detail::default_invoke_on(std::declval<Executor>(), std::declval<Invocable>(), std::declval<Args>()...));


} // end namespace detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../epilogue.hpp"

