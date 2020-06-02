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

#include "../../prologue.hpp"

#include <exception>
#include <type_traits>
#include <utility>
#include "../../functional/closure.hpp"
#include "../../type_traits/remove_cvref.hpp"
#include "../../../execution/executor/execute.hpp"
#include "../../../sender/connect.hpp"
#include "../../../sender/is_sender_to.hpp"
#include "../../../sender/is_typed_sender.hpp"
#include "../../../sender/sender_base.hpp"
#include "../../../sender/set_done.hpp"
#include "../../../sender/set_error.hpp"
#include "../../../sender/set_value.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


// XXX consider promoting this to go alongside set_value
struct try_set_value
{
  template<class R, class... Args>
  CUSEND_ANNOTATION
  void operator()(R&& r, Args&&... args) const
  {
#ifdef __CUDA_ARCH__
    set_value(std::move(r), std::move(args)...);
#else
    try
    {
      set_value(std::move(r), std::move(args)...);
    }
    catch(...)
    {
      set_error(std::move(r), std::current_exception());
    }
#endif
  }
};


// this functor accepts an invocable as its first parameter
// and arguments to that invocable as trailing parameters
// it invokes an rvalue reference to the invocable on the trailing parameters as rvalue references
struct move_and_invoke
{
  template<class F, class... Args>
  CUSEND_ANNOTATION
  auto operator()(F&& f, Args&&... args) const
    -> decltype(std::move(f)(std::move(args)...))
  {
    return std::move(f)(std::move(args)...);
  }
};


template<class Executor, class Receiver>
class via_receiver
{
  private:
    Executor ex_;
    Receiver receiver_;

  public:
    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    via_receiver(const Executor& ex, Receiver&& receiver)
      : ex_{ex},
        receiver_{std::move(receiver)}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    via_receiver(const via_receiver&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    via_receiver(via_receiver&&) = default;

    template<class... Args>
    CUSEND_ANNOTATION
    void set_value(Args&&... args) &&
    {
      // binding with move_and_invoke ensures that the receiver and arguments are moved() into try_set_value
      auto f = detail::bind(move_and_invoke{}, try_set_value{}, std::move(receiver_), std::forward<Args>(args)...);
      execution::execute(ex_, std::move(f));
    }

    template<class E>
    CUSEND_ANNOTATION
    void set_error(E&& error) && noexcept
    {
      // binding with move_and_invoke ensures that the receiver and arguments are moved() into set_error
      auto f = detail::bind(move_and_invoke{}, CUSEND_NAMESPACE::set_error, std::move(receiver_), std::forward<E>(error));
      execution::execute(ex_, std::move(f));
    }

    CUSEND_ANNOTATION
    void set_done() && noexcept
    {
      // binding with move_and_invoke ensures that the receiver and arguments are moved() into set_done
      auto f = detail::bind(move_and_invoke{}, CUSEND_NAMESPACE::set_done, std::move(receiver_));
      execution::execute(ex_, std::move(f));
    }
};


template<class Executor, class Receiver>
CUSEND_ANNOTATION
via_receiver<Executor,Receiver> make_via_receiver(const Executor& ex, Receiver&& r)
{
  return {ex, std::forward<Receiver>(r)};
}


// when via_sender's sender is untyped, via_sender should inherit from sender_base
template<class Sender, class Enable = void>
struct via_sender_base : cusend::sender_base {};

// when via_sender's sender is typed, via_sender should have nested sender traits
template<class Sender>
struct via_sender_base<Sender, typename std::enable_if<is_typed_sender<Sender>::value>::type>
{
  template<template<class...> class Tuple, template<class...> class Variant>
  using value_types = typename sender_traits<Sender>::template value_types<Tuple,Variant>;

  template<template<class...> class Variant>
  using error_types = typename sender_traits<Sender>::template error_types<Variant>;

  static constexpr bool sends_done = sender_traits<Sender>::sends_done;
};


template<class Sender, class Executor>
class via_sender : public via_sender_base<Sender>
{
  private:
    Sender predecessor_;
    Executor ex_;

  public:
    CUSEND_ANNOTATION
    via_sender(Sender&& predecessor, const Executor& ex)
      : predecessor_{std::move(predecessor)},
        ex_{ex}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    via_sender(const via_sender&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    via_sender(via_sender&&) = default;

    template<class Receiver,
             CUSEND_REQUIRES(is_sender_to<Sender&&,Receiver&&>::value)
            >
    CUSEND_ANNOTATION
    connect_t<Sender,via_receiver<Executor,remove_cvref_t<Receiver>>>
      connect(Receiver&& receiver) &&
    {
      return CUSEND_NAMESPACE::connect(std::move(predecessor_), make_via_receiver(std::move(ex_), std::move(receiver)));
    }
};


template<class Sender, class Executor>
CUSEND_ANNOTATION
via_sender<remove_cvref_t<Sender>, Executor> default_via(Sender&& predecessor, const Executor& ex)
{
  return {std::forward<Sender>(predecessor), ex};
}


template<class S, class E>
using default_via_t = decltype(detail::default_via(std::declval<S>(), std::declval<E>()));


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

