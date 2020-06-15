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

#include "../../detail/prologue.hpp"

#include <utility>
#include "../../detail/functional/invoke.hpp"
#include "../../detail/type_traits/decay.hpp"
#include "../../detail/type_traits/invoke_result.hpp"
#include "../../detail/type_traits/is_nothrow_invocable.hpp"
#include "../../detail/type_traits/is_nothrow_receiver_of.hpp"
#include "../receiver/discard_receiver.hpp"
#include "../receiver/is_receiver.hpp"
#include "../receiver/is_receiver_of.hpp"
#include "../receiver/set_done.hpp"
#include "../receiver/set_error.hpp"
#include "../receiver/set_value.hpp"
#include "../sender/connect.hpp"
#include "../sender/is_sender_to.hpp"
#include "../sender/is_typed_sender.hpp"
#include "../sender/sender_base.hpp"
#include "detail/type_list.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Receiver, class Function>
class transform_receiver
{
  public:
    CUSEND_EXEC_CHECK_DISABLE
    template<class R, class F,
             CUSEND_REQUIRES(std::is_constructible<Receiver,R&&>::value),
             CUSEND_REQUIRES(std::is_constructible<Function,F&&>::value)
            >
    CUSEND_ANNOTATION
    transform_receiver(R&& receiver, F&& f)
      : receiver_{std::forward<R>(receiver)},
        f_{std::forward<F>(f)}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    transform_receiver(const transform_receiver&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    transform_receiver(transform_receiver&&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    ~transform_receiver() = default;


    // Function returns void case
    template<class... Args, class Result = invoke_result_t<Function, Args&&...>,
             CUSEND_REQUIRES(std::is_void<Result>::value),
             CUSEND_REQUIRES(is_receiver_of<Receiver, Result>::value)
            >
    void set_value(Args&&... args) &&
      noexcept(is_nothrow_invocable<Function, Args...>::value and is_nothrow_receiver_of<Receiver>::value)
    {
      detail::invoke(std::move(f_), std::forward<Args>(args)...);
      CUSEND_NAMESPACE::set_value(std::move(receiver_));
    }

    // Function returns non-void case
    template<class... Args, class Result = invoke_result_t<Function, Args&&...>,
             CUSEND_REQUIRES(!std::is_void<Result>::value),
             CUSEND_REQUIRES(is_receiver_of<Receiver, Result>::value)
            >
    CUSEND_ANNOTATION
    void set_value(Args&&... args) &&
      noexcept(is_nothrow_invocable<Function, Args...>::value and is_nothrow_receiver_of<Receiver, Result>::value)
    {
      CUSEND_NAMESPACE::set_value(std::move(receiver_), detail::invoke(std::move(f_), std::forward<Args>(args)...));
    }

    template<class Error,
             CUSEND_REQUIRES(is_receiver<Receiver, Error>::value)
            >
    CUSEND_ANNOTATION
    void set_error(Error&& error) && noexcept
    {
      CUSEND_NAMESPACE::set_error(std::move(receiver_), std::forward<Error>(error));
    }

    CUSEND_ANNOTATION
    void set_done() && noexcept
    {
      CUSEND_NAMESPACE::set_done(std::move(receiver_));
    }

  private:
    Receiver receiver_;
    Function f_;
};


template<class Receiver, class Function>
CUSEND_ANNOTATION
transform_receiver<decay_t<Receiver>, decay_t<Function>> make_transform_receiver(Receiver&& receiver, Function&& continuation)
{
  return {std::forward<Receiver>(receiver), std::forward<Function>(continuation)};
}


namespace transform_sender_detail
{

// transform_sender has a lot of details because we need to ensure
// it is a "typed sender" when possible.
// That requires a lot of C++ metaprogramming, below.


// the purpose of this metafunction is to unpack a type list,
// interpret the types as a list of arguments,
// and evaluate invoke_result<F,Args...>
template<class F, class ArgList>
struct invoke_result_of_arg_list;

template<class F, class... Args>
struct invoke_result_of_arg_list<F, type_list<Args...>>
{
  using type = invoke_result_t<F,Args...>;
};

template<class F, class ArgList>
using invoke_result_of_arg_list_t = typename invoke_result_of_arg_list<F,ArgList>::type;


// the purpose of this meta function is to take a function type, and a list of argument lists,
// where each "inner" type list is interpreted as a list of arguments for F
//
// it returns
//
//   type_list<invoke_result_t<F,Args0...>, invoke_result_t<F,Args1...>...>
//
// for each argument list in ListOfArgLists
template<class F, class ListOfArgLists>
struct invoke_result_of_each_arg_list;

template<class F, class... ArgLists>
struct invoke_result_of_each_arg_list<F, type_list<ArgLists...>>
{
  using type = type_list<invoke_result_of_arg_list_t<F,ArgLists>...>;
};


// this metafunction takes a list of types and "wraps" each of them into a single-element Tuple
// the entire list of tuples is collected into a Variant
template<class TypeList, template<class...> class Variant, template<class...> class Tuple>
struct type_list_as_single_element_tuples;

template<class... Types, template<class...> class Variant, template<class...> class Tuple>
struct type_list_as_single_element_tuples<type_list<Types...>, Variant, Tuple>
{
  using type = Variant<Tuple<Types>...>;
};

template<class TypeList, template<class...> class Variant, template<class...> class Tuple>
using type_list_as_single_element_tuples_t = typename type_list_as_single_element_tuples<TypeList, Variant, Tuple>::type;


// the purpose of transform_sender_base is to inject sender traits
// into transform_sender when it is possible to do (i.e., Sender is typed)
template<class Sender, class Function, class Enable = void>
struct transform_sender_base : public sender_base {};

// If Sender is typed, define nested traits
template<class Sender, class Function>
struct transform_sender_base<Sender,Function, typename std::enable_if<is_typed_sender<Sender>::value>::type>
{
  private:
    using adaptee_traits = sender_traits<Sender>;

    // collect value types from the adaptee into type_lists
    using list_of_arg_lists = typename sender_traits<Sender>::template value_types<type_list,type_list>;

    // now, list_of_arg_lists should look like this:
    // list_of_arg_lists = type_list<type_list<A,B,C,...>, type_list<X,Y,Z,...>, ...>

    // apply invoke_result to each inner arg list
    using list_of_results = invoke_result_of_each_arg_list<Function, list_of_arg_lists>;

    // now, list_of_results should look like this:
    // list_of_results = type_list<R0, R1, R2...>

    // value_types needs to return
    // Variant<Tuple<R0>, Tuple<R1>, Tuple<R2>...>
    //
    // so, value_types needs to wrap each element of list_of_results with value_types' Tuple template parameter
    // and collect that list into value_types' Variant template parmeter

  public:
    template<template<class...> class Tuple, template<class...> class Variant>
    using value_types = type_list_as_single_element_tuples<list_of_results, Variant, Tuple>;

    template<template<class...> class Variant>
    using error_types = typename sender_traits<Sender>::template error_types<Variant>;

    constexpr static bool sends_done = sender_traits<Sender>::sends_done;
};


} // end transform_sender_detail


template<class Sender, class Function>
class transform_sender : public transform_sender_detail::transform_sender_base<Sender,Function>
{
  private:
    Sender predecessor_;
    Function continuation_;

  public:
    CUSEND_EXEC_CHECK_DISABLE
    template<class OtherSender, class OtherFunction,
             CUSEND_REQUIRES(std::is_constructible<Sender,OtherSender&&>::value),
             CUSEND_REQUIRES(std::is_constructible<Function,OtherFunction&&>::value)
            >
    CUSEND_ANNOTATION
    transform_sender(OtherSender&& predecessor, OtherFunction&& continuation)
      : predecessor_{std::forward<OtherSender>(predecessor)},
        continuation_{std::forward<OtherFunction>(continuation)}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    transform_sender(const transform_sender&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    transform_sender(transform_sender&&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    ~transform_sender() = default;

    template<class Receiver,
             CUSEND_REQUIRES(is_receiver<Receiver>::value),
             CUSEND_REQUIRES(is_sender_to<Sender, transform_receiver<Receiver, Function>>::value)
            >
    CUSEND_ANNOTATION
    auto connect(Receiver&& r) &&
      -> decltype(CUSEND_NAMESPACE::connect(std::move(predecessor_), detail::make_transform_receiver(std::move(r), std::move(continuation_))))
    {
      return CUSEND_NAMESPACE::connect(std::move(predecessor_), detail::make_transform_receiver(std::move(r), std::move(continuation_)));
    }

    // this overload allows makes transform_sender a "multi-shot" sender when both the predecessor and continuation are copyable
    // XXX should introduce is_multishot_sender or something
    template<class Receiver,
             CUSEND_REQUIRES(is_receiver<Receiver>::value),
             CUSEND_REQUIRES(is_sender_to<Sender, transform_receiver<Receiver, Function>>::value)
            >
    CUSEND_ANNOTATION
    auto connect(Receiver&& r) const &
      -> decltype(CUSEND_NAMESPACE::connect(predecessor_, detail::make_transform_receiver(std::move(r), continuation_)))
    {
      return CUSEND_NAMESPACE::connect(predecessor_, detail::make_transform_receiver(std::move(r), continuation_));
    }
};


template<class Sender, class Function,
         CUSEND_REQUIRES(is_sender_to<Sender, transform_receiver<discard_receiver, Function>>::value)
        >
CUSEND_ANNOTATION
detail::transform_sender<detail::decay_t<Sender>, detail::decay_t<Function>>
  default_transform(Sender&& predecessor, Function&& continuation)
{
  return {std::forward<Sender>(predecessor), std::forward<Function>(continuation)};
}


template<class S, class F>
using default_transform_t = decltype(detail::default_transform(std::declval<S>(), std::declval<F>()));


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

