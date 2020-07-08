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
#include "../../detail/static_const.hpp"
#include "../../detail/type_traits/is_detected.hpp"
#include "detail/default_bulk_schedule.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Scheduler, class Shape, class Sender>
using bulk_schedule_member_function_t = decltype(std::declval<Scheduler>().bulk_schedule(std::declval<Shape>(), std::declval<Sender>()));

template<class Scheduler, class Shape, class Sender>
using has_bulk_schedule_member_function = is_detected<bulk_schedule_member_function_t, Scheduler, Shape, Sender>;


template<class Scheduler, class Shape, class Sender>
using bulk_schedule_free_function_t = decltype(bulk_schedule(std::declval<Scheduler>(), std::declval<Shape>(), std::declval<Sender>()));

template<class Scheduler, class Shape, class Sender>
using has_bulk_schedule_free_function = is_detected<bulk_schedule_free_function_t, Scheduler, Shape, Sender>;


// this is the type of the bulk_schedule CPO
struct dispatch_bulk_schedule
{
  CUSEND_EXEC_CHECK_DISABLE
  template<class Scheduler, class Shape, class Sender,
           CUSEND_REQUIRES(has_bulk_schedule_member_function<Scheduler&&,Shape&&,Sender&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr bulk_schedule_member_function_t<Scheduler&&,Shape&&,Sender&&>
    operator()(Scheduler&& scheduler, Shape&& shape, Sender&& sender) const
  {
    return std::forward<Scheduler>(scheduler).bulk_schedule(std::forward<Shape>(shape), std::forward<Sender>(sender));
  }


  CUSEND_EXEC_CHECK_DISABLE
  template<class Scheduler, class Shape, class Sender,
           CUSEND_REQUIRES(!has_bulk_schedule_member_function<Scheduler&&,Shape&&,Sender&&>::value),
           CUSEND_REQUIRES(has_bulk_schedule_free_function<Scheduler&&,Shape&&,Sender&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr bulk_schedule_free_function_t<Scheduler&&,Shape&&,Sender&&>
    operator()(Scheduler&& scheduler, Shape&& shape, Sender&& sender) const
  {
    return bulk_schedule(std::forward<Scheduler>(scheduler), std::forward<Shape>(shape), std::forward<Sender>(sender));
  }


  CUSEND_EXEC_CHECK_DISABLE
  template<class Scheduler, class Shape, class Sender,
           CUSEND_REQUIRES(!has_bulk_schedule_member_function<Scheduler&&,Shape&&,Sender&&>::value),
           CUSEND_REQUIRES(!has_bulk_schedule_free_function<Scheduler&&,Shape&&,Sender&&>::value),
           CUSEND_REQUIRES(is_detected<default_bulk_schedule_t,Scheduler&&,Shape&&,Sender&&>::value)
          >
  CUSEND_ANNOTATION
  constexpr default_bulk_schedule_t<Scheduler&&,Shape&&,Sender&&>
    operator()(Scheduler&& scheduler, Shape&& shape, Sender&& sender) const
  {
    return detail::default_bulk_schedule(std::forward<Scheduler>(scheduler), std::forward<Shape>(shape), std::forward<Sender>(sender));
  }
};


} // end detail


namespace
{


// define the bulk_schedule customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& bulk_schedule = detail::static_const<detail::dispatch_bulk_schedule>::value;
#else
const __device__ detail::dispatch_bulk_schedule bulk_schedule;
#endif


} // end anonymous namespace


template<class Scheduler, class Shape, class Sender>
using bulk_schedule_t = decltype(bulk_schedule(std::declval<Scheduler>(), std::declval<Shape>(), std::declval<Sender>()));


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../../detail/epilogue.hpp"

