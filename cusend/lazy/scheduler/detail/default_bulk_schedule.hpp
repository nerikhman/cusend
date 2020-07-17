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

#include "../../../detail/prologue.hpp"

#include <type_traits>
#include <utility>
#include "../../../detail/type_traits/remove_cvref.hpp"
#include "../../connect.hpp"
#include "../../sender/is_typed_sender.hpp"
#include "../get_executor.hpp"
#include "../is_scheduler.hpp"
#include "../scheduler_coordinate.hpp"
#include "detail/fan_out_receiver.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Scheduler, class TypedSender>
class bulk_sender
{
  private:
    using executor_type = get_executor_t<Scheduler>;
    using coordinate_type = scheduler_coordinate_t<Scheduler>;

    Scheduler scheduler_;
    coordinate_type shape_;
    TypedSender predecessor_;

  public:
    template<template<class...> class Tuple, template<class...> class Variant>
    using value_types = typename sender_traits<TypedSender>::template value_types<Tuple, Variant>;

    template<template<class...> class Variant>
    using error_types = typename sender_traits<TypedSender>::template error_types<Variant>;

    static constexpr bool sends_done = sender_traits<TypedSender>::sends_done;


    CUSEND_EXEC_CHECK_DISABLE
    template<class Sender>
    CUSEND_ANNOTATION
    bulk_sender(const Scheduler& scheduler, coordinate_type shape, Sender&& predecessor)
      : scheduler_{scheduler},
        shape_{shape},
        predecessor_{std::forward<Sender>(predecessor)}
    {}


    CUSEND_ANNOTATION
    executor_type executor() const
    {
      return get_executor(scheduler_);
    }


    template<class ManyReceiver,
             CUSEND_REQUIRES(can_make_fan_out_receiver<TypedSender,executor_type,coordinate_type,ManyReceiver&&>::value)
            >
    CUSEND_ANNOTATION
    auto connect(ManyReceiver&& r) &&
      -> decltype(CUSEND_NAMESPACE::connect(std::move(predecessor_), detail::make_fan_out_receiver<TypedSender>(executor(), shape_, std::forward<ManyReceiver>(r))))
    {
      return CUSEND_NAMESPACE::connect(std::move(predecessor_), detail::make_fan_out_receiver<TypedSender>(executor(), shape_, std::forward<ManyReceiver>(r)));
    }


    template<class OtherScheduler,
             CUSEND_REQUIRES(is_scheduler<OtherScheduler>::value)
            >
    CUSEND_ANNOTATION
    bulk_sender<OtherScheduler,TypedSender> via(const OtherScheduler& scheduler) &&
    {
      return {scheduler, shape_, std::move(predecessor_)};
    }


    template<class OtherScheduler,
             CUSEND_REQUIRES(is_scheduler<OtherScheduler>::value),
             CUSEND_REQUIRES(std::is_copy_constructible<TypedSender>::value)
            >
    CUSEND_ANNOTATION
    bulk_sender<OtherScheduler,TypedSender> via(const OtherScheduler& scheduler) const
    {
      return {scheduler, shape_, predecessor_};
    }
};


template<class Scheduler, class TypedSender,
         CUSEND_REQUIRES(is_scheduler<Scheduler>::value),
         CUSEND_REQUIRES(is_typed_sender<TypedSender&&>::value)
        >
CUSEND_ANNOTATION
bulk_sender<Scheduler, remove_cvref_t<TypedSender>> default_bulk_schedule(const Scheduler& scheduler, scheduler_coordinate_t<Scheduler> shape, TypedSender&& sender)
{
  return {scheduler, shape, std::forward<TypedSender>(sender)};
}


template<class Scheduler, class Shape, class Sender>
using default_bulk_schedule_t = decltype(detail::default_bulk_schedule(std::declval<Scheduler>(), std::declval<Shape>(), std::declval<Sender>()));


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../../../detail/epilogue.hpp"

