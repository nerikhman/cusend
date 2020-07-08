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
#include "../../execution/executor/is_executor.hpp"
#include "../is_scheduler.hpp"
#include "../submit.hpp"
#include "invocable_as_receiver.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Executor,
         CUSEND_REQUIRES(execution::is_executor<Executor>::value)
        >
CUSEND_ANNOTATION
Executor default_get_executor(const Executor& executor)
{
  return executor;
}


template<class Scheduler>
class scheduler_as_executor
{
  public:
    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    scheduler_as_executor(const Scheduler& scheduler)
      : scheduler_{scheduler}
    {}


    CUSEND_EXEC_CHECK_DISABLE
    scheduler_as_executor(const scheduler_as_executor&) = default;


    template<class Function,
             CUSEND_REQUIRES(detail::is_invocable<Function>::value)
            >
    CUSEND_ANNOTATION
    void execute(Function&& f) const
    {
      submit(schedule(scheduler_), detail::as_receiver(std::forward<Function>(f)));
    }

  private:
    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    friend bool operator==(const scheduler_as_executor& lhs, const scheduler_as_executor& rhs)
    {
      return lhs.scheduler_ == rhs.scheduler_;
    }


    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    friend bool operator!=(const scheduler_as_executor& lhs, const scheduler_as_executor& rhs)
    {
      return lhs.scheduler_ != rhs.scheduler_;
    }


    Scheduler scheduler_;
};


template<class Scheduler,
         CUSEND_REQUIRES(!execution::is_executor<Scheduler>::value),
         CUSEND_REQUIRES(is_scheduler<Scheduler>::value)
        >
CUSEND_ANNOTATION
scheduler_as_executor<Scheduler> default_get_executor(const Scheduler& ex)
{
  return {ex};
}


template<class T>
using default_get_executor_t = decltype(detail::default_get_executor(std::declval<T>()));


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../../detail/epilogue.hpp"

