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
#include "../../../execution/executor/is_executor.hpp"
#include "detail/executor_as_sender.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Executor>
class executor_as_scheduler
{
  public:
    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    executor_as_scheduler(const Executor& executor)
      : executor_{executor}
    {}


    CUSEND_EXEC_CHECK_DISABLE
    executor_as_scheduler(const executor_as_scheduler&) = default;


    CUSEND_ANNOTATION
    const Executor& executor() const
    {
      return executor_;
    }


    CUSEND_ANNOTATION
    executor_as_sender<Executor> schedule() const
    {
      return {executor()};
    }


  private:
    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    friend bool operator==(const executor_as_scheduler& lhs, const executor_as_scheduler& rhs)
    {
      return lhs.executor_ == rhs.executor_;
    }


    CUSEND_EXEC_CHECK_DISABLE
    CUSEND_ANNOTATION
    friend bool operator!=(const executor_as_scheduler& lhs, const executor_as_scheduler& rhs)
    {
      return lhs.executor_ != rhs.executor_;
    }


    Executor executor_;
};


template<class Executor,
         CUSEND_REQUIRES(execution::is_executor<Executor>::value)
        >
CUSEND_ANNOTATION
executor_as_scheduler<Executor> default_as_scheduler(const Executor& ex)
{
  return {ex};
}


template<class E>
using default_as_scheduler_t = decltype(detail::default_as_scheduler(std::declval<E>()));


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../../../detail/epilogue.hpp"

