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

#include <utility>
#include "../../../discard_receiver.hpp"
#include "../../../sender/connect.hpp"
#include "../../../sender/is_receiver.hpp"
#include "../../../sender/is_receiver_of.hpp"
#include "../../../sender/is_sender_to.hpp"
#include "../../../sender/sender_base.hpp"
#include "../../../sender/set_done.hpp"
#include "../../../sender/set_error.hpp"
#include "../../../sender/set_value.hpp"
#include "../../functional/invoke.hpp"
#include "../../type_traits/decay.hpp"
#include "../../type_traits/invoke_result.hpp"
#include "../../type_traits/is_nothrow_invocable.hpp"
#include "../../type_traits/is_nothrow_receiver_of.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Receiver, class Function>
class then_receiver
{
  public:
    CUSEND_EXEC_CHECK_DISABLE
    template<class R, class F,
             CUSEND_REQUIRES(std::is_constructible<Receiver,R&&>::value),
             CUSEND_REQUIRES(std::is_constructible<Function,F&&>::value)
            >
    CUSEND_ANNOTATION
    then_receiver(R&& receiver, F&& f)
      : receiver_{std::forward<R>(receiver)},
        f_{std::forward<F>(f)}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    then_receiver(then_receiver&& other) noexcept
      : receiver_{std::move(other.receiver_)},
        f_{std::move(other.f_)}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    ~then_receiver() noexcept {}


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
then_receiver<decay_t<Receiver>, decay_t<Function>> make_then_receiver(Receiver&& receiver, Function&& continuation)
{
  return {std::forward<Receiver>(receiver), std::forward<Function>(continuation)};
}


template<class Sender, class Function>
class then_sender : public sender_base
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
    then_sender(OtherSender&& predecessor, OtherFunction&& continuation)
      : predecessor_{std::forward<OtherSender>(predecessor)},
        continuation_{std::forward<OtherFunction>(continuation)}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    then_sender(const then_sender&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    then_sender(then_sender&&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    ~then_sender() = default;

    template<class Receiver,
             CUSEND_REQUIRES(is_receiver<Receiver>::value),
             CUSEND_REQUIRES(is_sender_to<Sender, then_receiver<Receiver, Function>>::value)
            >
    CUSEND_ANNOTATION
    auto connect(Receiver&& r) &&
      -> decltype(CUSEND_NAMESPACE::connect(std::move(predecessor_), detail::make_then_receiver(std::move(r), std::move(continuation_))))
    {
      return CUSEND_NAMESPACE::connect(std::move(predecessor_), detail::make_then_receiver(std::move(r), std::move(continuation_)));
    }

    // this overload allows makes then_sender a "multi-shot" sender when both the predecessor and continuation are copyable
    // XXX should introduce is_multishot_sender or something
    template<class Receiver,
             CUSEND_REQUIRES(is_receiver<Receiver>::value),
             CUSEND_REQUIRES(is_sender_to<Sender, then_receiver<Receiver, Function>>::value)
            >
    CUSEND_ANNOTATION
    auto connect(Receiver&& r) const &
      -> decltype(CUSEND_NAMESPACE::connect(predecessor_, detail::make_then_receiver(std::move(r), continuation_)))
    {
      return CUSEND_NAMESPACE::connect(predecessor_, detail::make_then_receiver(std::move(r), continuation_));
    }
};


template<class Sender, class Function,
         CUSEND_REQUIRES(is_sender_to<Sender, then_receiver<discard_receiver, Function>>::value)
        >
CUSEND_ANNOTATION
detail::then_sender<detail::decay_t<Sender>, detail::decay_t<Function>>
  default_then(Sender&& predecessor, Function&& continuation)
{
  return {std::forward<Sender>(predecessor), std::forward<Function>(continuation)};
}


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

