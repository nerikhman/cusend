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

#include <type_traits>
#include <utility>
#include "detail/combinators/on/dispatch_on.hpp"
#include "detail/combinators/transform.hpp"
#include "detail/combinators/via/dispatch_via.hpp"
#include "get_executor.hpp"
#include "sender/connect.hpp"
#include "sender/is_sender.hpp"
#include "sender/sender_traits.hpp"
#include "sender/submit.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


// this template wraps another sender and introduces convenient chaining via member functions
template<class Sender>
class chaining_sender
{
  private:
    Sender sender_;

  public:
    CUSEND_EXEC_CHECK_DISABLE
    template<class OtherSender,
             CUSEND_REQUIRES(is_sender<OtherSender&&>::value),
             CUSEND_REQUIRES(std::is_constructible<Sender,OtherSender&&>::value)
            >
    CUSEND_ANNOTATION
    chaining_sender(OtherSender&& sender)
      : sender_{std::forward<OtherSender>(sender)}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    chaining_sender(const chaining_sender& other)
      : sender_{other.sender_}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    chaining_sender(chaining_sender&& other)
      : sender_{std::move(other.sender_)}
    {}


    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    ~chaining_sender() {}


    template<class Receiver,
             CUSEND_REQUIRES(is_sender_to<Sender&&,Receiver&&>::value)
            >
    CUSEND_ANNOTATION
    auto connect(Receiver&& receiver) &&
      -> decltype(CUSEND_NAMESPACE::connect(std::move(sender_), std::forward<Receiver>(receiver)))
    {
      return CUSEND_NAMESPACE::connect(std::move(sender_), std::forward<Receiver>(receiver));
    }


    // the use of a defaulted template parameter allows SFINAE to kick in
    template<class S = Sender,
             CUSEND_REQUIRES(detail::is_detected<get_executor_t,S>::value)>
    CUSEND_ANNOTATION
    get_executor_t<S> executor() const
    {
      return CUSEND_NAMESPACE::get_executor(sender_);
    }


    template<class Executor,
             CUSEND_REQUIRES(detail::can_dispatch_on<Sender&&,const Executor&>::value)
            >
    CUSEND_ANNOTATION
    chaining_sender<detail::dispatch_on_t<Sender&&,const Executor&>>
      on(const Executor& ex) &&
    {
      return {detail::dispatch_on(std::move(sender_), ex)};
    }


    template<class Receiver,
             CUSEND_REQUIRES(is_sender_to<Sender&&,Receiver&&>::value)
            >
    CUSEND_ANNOTATION
    void submit(Receiver&& receiver) &&
    {
      CUSEND_NAMESPACE::submit(std::move(sender_), std::forward<Receiver>(receiver));
    }


    template<class Function,
             CUSEND_REQUIRES(detail::is_detected<detail::transform_t,Sender&&,Function&&>::value)
            >
    CUSEND_ANNOTATION
    chaining_sender<detail::transform_t<Sender&&,Function&&>>
      transform(Function&& continuation) &&
    {
      return {detail::transform(std::move(sender_), std::forward<Function>(continuation))};
    }


    template<class Executor,
             CUSEND_REQUIRES(detail::can_dispatch_via<Sender&&,const Executor&>::value)
            >
    CUSEND_ANNOTATION
    chaining_sender<detail::dispatch_via_t<Sender&&,const Executor&>>
      via(const Executor& ex) &&
    {
      return {detail::dispatch_via(std::move(sender_), ex)};
    }
};


// specialize sender_traits
template<class Sender>
struct sender_traits<chaining_sender<Sender>> : sender_traits<Sender> {};


// this utility allows clients (such as the sender combinator CPOs) to ensure that the senders they return
// aren't multiply-wrapped chaining_senders
// i.e., chaining_sender<chaining_sender<...>> is unhelpful, so let's avoid creating those
template<class Sender,
         CUSEND_REQUIRES(is_sender<Sender&&>::value),
         CUSEND_REQUIRES(std::is_rvalue_reference<Sender&&>::value)
        >
CUSEND_ANNOTATION
chaining_sender<detail::decay_t<Sender>> ensure_chaining_sender(Sender&& sender)
{
  return {std::move(sender)};
}

template<class Sender>
CUSEND_ANNOTATION
chaining_sender<Sender> ensure_chaining_sender(chaining_sender<Sender>&& sender)
{
  return std::move(sender);
}


template<class Sender>
using ensure_chaining_sender_t = decltype(ensure_chaining_sender(std::declval<Sender>()));


// make multiply-wrapped chaining_senders illegal for now 
// XXX eliminate this ASAP, there might be some use case for multiple-wrapping
template<class Sender>
class chaining_sender<chaining_sender<Sender>>;


CUSEND_NAMESPACE_CLOSE_BRACE

#include "detail/epilogue.hpp"

