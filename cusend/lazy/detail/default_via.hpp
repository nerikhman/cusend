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

#include <exception>
#include <type_traits>
#include <utility>
#include "../../detail/type_traits/remove_cvref.hpp"
#include "../just_on.hpp"
#include "../receiver/set_done.hpp"
#include "../receiver/set_error.hpp"
#include "../receiver/set_value.hpp"
#include "../sender/connect.hpp"
#include "../sender/is_sender_to.hpp"
#include "../sender/is_typed_sender.hpp"
#include "../sender/sender_base.hpp"
#include "../sender/submit.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Scheduler, class Receiver>
class via_receiver
{
  private:
    Scheduler scheduler_;
    Receiver receiver_;

  public:
    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    via_receiver(const Scheduler& scheduler, Receiver&& receiver)
      : scheduler_{scheduler},
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
      submit(just_on(scheduler_, std::forward<Args>(args)...), std::move(receiver_));
    }

    template<class E>
    CUSEND_ANNOTATION
    void set_error(E&& error) && noexcept
    {
      CUSEND_NAMESPACE::set_error(std::move(receiver_), std::forward<E>(error));
    }

    CUSEND_ANNOTATION
    void set_done() && noexcept
    {
      CUSEND_NAMESPACE::set_done(std::move(receiver_));
    }
};


template<class Scheduler, class Receiver>
CUSEND_ANNOTATION
via_receiver<Scheduler,Receiver> make_via_receiver(const Scheduler& scheduler, Receiver&& r)
{
  return {scheduler, std::forward<Receiver>(r)};
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


template<class Sender, class Scheduler>
class via_sender : public via_sender_base<Sender>
{
  public:
    CUSEND_ANNOTATION
    via_sender(Sender&& sender, const Scheduler& scheduler)
      : sender_{std::move(sender)},
        scheduler_{scheduler}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    via_sender(const via_sender&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    via_sender(via_sender&&) = default;


    template<class Receiver,
             CUSEND_REQUIRES(is_sender_to<Sender&&,Receiver&&>::value)
            >
    CUSEND_ANNOTATION
    connect_t<Sender,via_receiver<Scheduler,remove_cvref_t<Receiver>>>
      connect(Receiver&& receiver) &&
    {
      return CUSEND_NAMESPACE::connect(std::move(sender_), make_via_receiver(scheduler_, std::move(receiver)));
    }


    template<class OtherScheduler,
             CUSEND_REQUIRES(is_scheduler<OtherScheduler>::value)
            >
    CUSEND_ANNOTATION
    via_sender<Sender, OtherScheduler> on(const OtherScheduler& scheduler) &&
    {
      return {std::move(sender_), scheduler};
    }


    template<class OtherScheduler,
             CUSEND_REQUIRES(is_scheduler<OtherScheduler>::value),
             CUSEND_REQUIRES(std::is_copy_constructible<Sender>::value)
            >
    CUSEND_ANNOTATION
    via_sender<Sender, OtherScheduler> on(const OtherScheduler& scheduler) const &
    {
      return {sender_, scheduler};
    }


  private:
    Sender sender_;
    Scheduler scheduler_;
};


template<class Sender, class Scheduler>
CUSEND_ANNOTATION
via_sender<remove_cvref_t<Sender>, Scheduler> default_via(Sender&& sender, const Scheduler& scheduler)
{
  return {std::forward<Sender>(sender), scheduler};
}


template<class Sender, class Scheduler>
using default_via_t = decltype(detail::default_via(std::declval<Sender>(), std::declval<Scheduler>()));


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

