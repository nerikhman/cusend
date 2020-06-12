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

#include <utility>
#include "../../is_scheduler.hpp"
#include "../../schedule.hpp"
#include "../../sender/connect.hpp"
#include "../../sender/is_receiver_of.hpp"
#include "../../sender/set_done.hpp"
#include "../../sender/set_error.hpp"
#include "../../sender/set_value.hpp"
#include "../functional/apply.hpp"
#include "../tuple.hpp"
#include "../type_traits/remove_cvref.hpp"
#include "../utility/index_sequence.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Receiver, class... Values>
class just_receiver
{
  private:
    detail::tuple<Receiver, Values...> receiver_and_values_;

    CUSEND_ANNOTATION
    Receiver& receiver()
    {
      return detail::get<0>(receiver_and_values_);
    }

  public:
    CUSEND_ANNOTATION
    just_receiver(Receiver&& receiver, Values&&... values)
      : receiver_and_values_{std::move(receiver), std::move(values)...}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    just_receiver(const just_receiver&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    just_receiver(just_receiver&&) = default;


    template<class R = Receiver,
             CUSEND_REQUIRES(is_receiver_of<R&&,Values&&...>::value)
            >
    CUSEND_ANNOTATION
    void set_value() &&
    {
      // note that this overload of set_value calls set_value on rvalue references to receiver_and_values_
      detail::apply(CUSEND_NAMESPACE::set_value, std::move(receiver_and_values_));
    }


    template<class E>
    CUSEND_ANNOTATION
    void set_error(E&& e) && noexcept
    {
      CUSEND_NAMESPACE::set_error(std::move(receiver()), std::forward<E>(e));
    }


    CUSEND_ANNOTATION
    void set_done() && noexcept
    {
      CUSEND_NAMESPACE::set_done(std::move(receiver()));
    }
};


template<class Receiver, class... Values, std::size_t... I>
CUSEND_ANNOTATION
just_receiver<remove_cvref_t<Receiver>, Values...> make_just_receiver_impl(index_sequence<I...>, Receiver&& receiver, detail::tuple<Values...>&& values)
{
  return {std::forward<Receiver>(receiver), detail::get<I>(std::move(values))...};
}


template<class Receiver, class... Values>
CUSEND_ANNOTATION
just_receiver<remove_cvref_t<Receiver>, Values...> make_just_receiver(Receiver&& receiver, detail::tuple<Values...>&& values)
{
  return detail::make_just_receiver_impl(index_sequence_for<Values...>{}, std::forward<Receiver>(receiver), std::move(values));
}


template<class Scheduler, class... Values>
class just_sender
{
  private:
    Scheduler scheduler_;
    detail::tuple<Values...> values_;

  public:
    template<template<class...> class Tuple, template<class...> class Variant>
    using value_types = Variant<Tuple<Values...>>;

    template<template<class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    constexpr static bool sends_done = true;

    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    just_sender(const Scheduler& scheduler, detail::tuple<Values...>&& values)
      : scheduler_{scheduler},
        values_{std::move(values)}
    {}

    template<class... OtherValues>
    CUSEND_ANNOTATION
    just_sender(const Scheduler& scheduler, OtherValues&&... values)
      : just_sender{scheduler, detail::make_tuple(std::forward<OtherValues>(values)...)}
    {}

    CUSEND_EXEC_CHECK_DISABLE
    just_sender(const just_sender&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    just_sender(just_sender&&) = default;

    template<class Receiver,
             CUSEND_REQUIRES(is_receiver_of<Receiver, Values...>::value)
            >
    CUSEND_ANNOTATION
    auto connect(Receiver&& r) &&
      -> decltype(CUSEND_NAMESPACE::connect(schedule(scheduler_), detail::make_just_receiver(std::forward<Receiver>(r), std::move(values_))))
    {
      return CUSEND_NAMESPACE::connect(schedule(scheduler_), detail::make_just_receiver(std::forward<Receiver>(r), std::move(values_)));
    }

    template<class Receiver,
             CUSEND_REQUIRES(is_receiver_of<Receiver, Values...>::value)
            >
    CUSEND_ANNOTATION
    auto connect(Receiver&& r) const &
      -> decltype(CUSEND_NAMESPACE::connect(schedule(scheduler_), detail::make_just_receiver(std::forward<Receiver>(r), values_)))
    {
      return CUSEND_NAMESPACE::connect(scheduler(scheduler_), detail::make_just_receiver(std::forward<Receiver>(r), values_));
    }


    template<class OtherScheduler,
             CUSEND_REQUIRES(is_scheduler<OtherScheduler>::value),
             CUSEND_REQUIRES(std::is_copy_constructible<detail::tuple<Values...>>::value)
            >
    CUSEND_ANNOTATION
    just_sender<OtherScheduler,Values...> on(const OtherScheduler& scheduler) const
    {
      return {scheduler, values_};
    }


    template<class OtherScheduler,
             CUSEND_REQUIRES(is_scheduler<OtherScheduler>::value)
            >
    CUSEND_ANNOTATION
    just_sender<OtherScheduler,Values...> on(const OtherScheduler& scheduler) const
    {
      return {scheduler, std::move(values_)};
    }
};


template<class Scheduler, class... Values,
         CUSEND_REQUIRES(is_scheduler<Scheduler>::value)
        >
CUSEND_ANNOTATION
just_sender<Scheduler, remove_cvref_t<Values>...> default_just_on(const Scheduler& scheduler, Values&&... values)
{
  return {scheduler, std::forward<Values>(values)...};
}


template<class Scheduler, class... Values>
using default_just_on_t = decltype(detail::default_just_on(std::declval<Scheduler>(), std::declval<Values>()...));


} // end namespace detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../epilogue.hpp"

