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

#include <type_traits>
#include <utility>
#include "../../detail/type_traits/remove_cvref.hpp"
#include "../is_scheduler.hpp"
#include "../receiver/set_done.hpp"
#include "../receiver/set_error.hpp"
#include "../receiver/set_value.hpp"
#include "../connect.hpp"
#include "../is_sender_to.hpp"
#include "../schedule.hpp"
#include "../sender/is_sender.hpp"
#include "../sender/sender_base.hpp"
#include "../submit.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


// on_receiver calls submit(sender,receiver) inside set_value
// thus ensuring that the sender is submitted on the execution
// context associated with the originating on_sender
template<class Sender, class Receiver>
class on_receiver
{
  public:
    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    on_receiver(Sender&& sender, Receiver&& receiver)
      : sender_{std::move(sender)},
        receiver_{std::move(receiver)}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    on_receiver(const on_receiver&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    on_receiver(on_receiver&&) = default;


    CUSEND_ANNOTATION
    void set_value() &&
    {
      submit(std::move(sender_), std::move(receiver_));
    }


    template<class E>
    void set_error(E&& e) && noexcept
    {
      CUSEND_NAMESPACE::set_error(std::move(receiver_), std::forward<E>(e));
    }


    CUSEND_ANNOTATION
    void set_done() && noexcept
    {
      CUSEND_NAMESPACE::set_done(std::move(receiver_));
    }

  private:
    Sender sender_;
    Receiver receiver_;
};


template<class Sender, class Receiver>
CUSEND_ANNOTATION
on_receiver<remove_cvref_t<Sender>, remove_cvref_t<Receiver>> make_on_receiver(Sender&& sender, Receiver&& receiver)
{
  return {std::forward<Sender>(sender), std::forward<Receiver>(receiver)};
}


// XXX instead of deriving from sender_base,
//     this sender should send Sender's types, if any
template<class Sender, class Scheduler>
class on_sender : public sender_base
{
  public:
    CUSEND_EXEC_CHECK_DISABLE
    template<class OtherSender,
             CUSEND_REQUIRES(std::is_constructible<Sender,OtherSender&&>::value)
            >
    CUSEND_ANNOTATION
    on_sender(OtherSender&& sender, const Scheduler& scheduler)
      : sender_(std::forward<OtherSender>(sender)),
        scheduler_(scheduler)
    {}

    on_sender(const on_sender&) = default;

    on_sender(on_sender&&) = default;


    template<class Receiver,
             CUSEND_REQUIRES(is_sender_to<Sender&&,Receiver&&>::value)
            >
    CUSEND_ANNOTATION
    connect_t<schedule_t<Scheduler>, on_receiver<Sender, remove_cvref_t<Receiver>>>
      connect(Receiver&& r) &&
    {
      return CUSEND_NAMESPACE::connect(schedule(scheduler_), detail::make_on_receiver(std::move(sender_), std::forward<Receiver>(r)));
    }


    template<class OtherScheduler,
             CUSEND_REQUIRES(is_scheduler<OtherScheduler>::value)
            >
    CUSEND_ANNOTATION
    on_sender<Sender, OtherScheduler> on(const OtherScheduler& scheduler) &&
    {
      return {std::move(sender_), scheduler};
    }


    template<class OtherScheduler,
             CUSEND_REQUIRES(is_scheduler<OtherScheduler>::value),
             CUSEND_REQUIRES(std::is_copy_constructible<Sender>::value)
            >
    CUSEND_ANNOTATION
    on_sender<Sender, OtherScheduler> on(const OtherScheduler& scheduler) const &
    {
      return {sender_, scheduler};
    }


  private:
    Sender sender_;
    Scheduler scheduler_;
};


template<class Sender, class Scheduler,
         CUSEND_REQUIRES(is_sender<Sender>::value),
         CUSEND_REQUIRES(is_scheduler<Scheduler>::value)
        >
CUSEND_ANNOTATION
on_sender<remove_cvref_t<Sender>, Scheduler> default_on(Sender&& s, const Scheduler& scheduler)
{
  return {std::forward<Sender>(s), scheduler};
}


template<class Sender, class Scheduler>
using default_on_t = decltype(detail::default_on(std::declval<Sender>(), std::declval<Scheduler>()));


} // end namespace detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../../detail/epilogue.hpp"

