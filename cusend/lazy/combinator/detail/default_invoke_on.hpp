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

#include <utility>
#include "../../../detail/type_traits/is_invocable.hpp"
#include "../../scheduler/is_scheduler.hpp"
#include "../just_on.hpp"
#include "../transform.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Scheduler, class Invocable, class... Args,
         CUSEND_REQUIRES(is_scheduler<Scheduler>::value),
         CUSEND_REQUIRES(detail::is_invocable<Invocable,Args...>::value)
        >
CUSEND_ANNOTATION
auto default_invoke_on(const Scheduler& scheduler, Invocable&& f, Args&&... args)
  -> decltype(transform(just_on(scheduler, std::forward<Args>(args)...), std::forward<Invocable>(f)))
{
  return transform(just_on(scheduler, std::forward<Args>(args)...), std::forward<Invocable>(f));
}


template<class Scheduler, class Invocable, class... Args>
using default_invoke_on_t = decltype(detail::default_invoke_on(std::declval<Scheduler>(), std::declval<Invocable>(), std::declval<Args>()...));


} // end namespace detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../../../detail/epilogue.hpp"

