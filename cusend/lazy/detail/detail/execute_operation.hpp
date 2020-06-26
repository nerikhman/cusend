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
#include "../../../execution/executor/execute.hpp"
#include "../../../execution/executor/is_executor.hpp"
#include "../../../execution/executor/is_executor_of.hpp"
#include "../../receiver/is_receiver_of.hpp"
#include "indirect_receiver_as_invocable.hpp"
#include "receiver_as_invocable.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Executor, class Receiver>
class execute_operation
{
  public:
    CUSEND_EXEC_CHECK_DISABLE
    template<class OtherReceiver,
             CUSEND_REQUIRES(std::is_constructible<Receiver,OtherReceiver&&>::value)
            >
    CUSEND_ANNOTATION
    execute_operation(const Executor& ex, OtherReceiver&& receiver)
      : ex_(ex), receiver_(std::forward<OtherReceiver>(receiver))
    {}

    CUSEND_EXEC_CHECK_DISABLE
    execute_operation(const execute_operation&) = default;

    CUSEND_EXEC_CHECK_DISABLE
    execute_operation(execute_operation&&) = default;

#if CUSEND_HAS_EXCEPTIONS
    // copyable receiver case
    template<class R = Receiver,
             CUSEND_REQUIRES(std::is_copy_constructible<R>::value),
             CUSEND_REQUIRES(execution::is_executor_of<Executor,receiver_as_invocable<R>>::value)
            >
    CUSEND_ANNOTATION
    void start() noexcept
    {
      // if the receiver is copyable, copy it when creating the invocable to execute
      // we do this because the executor's memory may be in a different place than here,
      // and we'd rather not access the receiver indirectly though memory if we don't need to

      try
      {
        execution::execute(ex_, detail::copy_as_invocable(receiver_));
      }
      catch(...)
      {
        set_error(std::move(receiver_), std::current_exception());
      }
    }

    // non-copyable receiver case
    template<class R = Receiver,
             CUSEND_REQUIRES(!std::is_copy_constructible<R>::value),
             CUSEND_REQUIRES(execution::is_executor_of<Executor,indirect_receiver_as_invocable<R>>::value)
            >
    CUSEND_ANNOTATION
    void start() noexcept
    {
      try
      {
        execution::execute(ex_, detail::indirectly_as_invocable(&receiver_));
      }
      catch(...)
      {
        set_error(std::move(receiver_), std::current_exception());
      }
    }
#else
    template<class R = Receiver,
             CUSEND_REQUIRES(execution::is_executor_of<Executor,receiver_as_invocable<R>>::value)
            >
    CUSEND_ANNOTATION
    void start() noexcept
    {
      execution::execute(ex_, detail::move_as_invocable(receiver_));
    }
#endif

  private:
    Executor ex_;
    Receiver receiver_;
};


template<class Executor, class Receiver,
         CUSEND_REQUIRES(execution::is_executor<Executor>::value),
         CUSEND_REQUIRES(is_receiver_of<Receiver,void>::value)
        >
CUSEND_ANNOTATION
execute_operation<Executor, remove_cvref_t<Receiver>> make_execute_operation(const Executor& ex, Receiver&& r)
{
  return {ex, std::forward<Receiver>(r)};
}


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE


#include "../../../detail/epilogue.hpp"

