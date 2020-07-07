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

#include <future>
#include <type_traits>
#include "../../detail/event.hpp"
#include "../../detail/is_stream_executor.hpp"
#include "../../detail/type_traits/is_invocable.hpp"
#include "../../execution/executor/callback_executor.hpp"
#include "../../execution/executor/execute.hpp"
#include "../../lazy/detail/receiver_as_trivially_copyable_invocable.hpp"
#include "../../lazy/receiver/is_receiver_of.hpp"
#include "stream_of.hpp"
#include "stream_wait_for.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T, class Function,
         CUSEND_REQUIRES(is_invocable<Function&&,T&&>::value),
         CUSEND_REQUIRES(std::is_void<invoke_result_t<Function&&,T&&>>::value)
        >
detail::event then_execute(const execution::callback_executor& ex, std::future<T>&& predecessor, Function&& f)
{
  ex.execute([fut = std::move(predecessor), f = std::move(f)]() mutable
  {
    f(fut.get());
  });

  // return an event recorded on ex's stream
  return {ex.stream()};
}


template<class Function,
         CUSEND_REQUIRES(is_invocable<Function&&>::value),
         CUSEND_REQUIRES(std::is_void<invoke_result_t<Function&&>>::value)
        >
detail::event then_execute(const execution::callback_executor& ex, std::future<void>&& predecessor, Function&& f)
{
  ex.execute([fut = std::move(predecessor), f = std::move(f)]() mutable
  {
    fut.get();
    f();
  });

  // return an event recorded on ex's stream
  return {ex.stream()};
}


template<class StreamExecutor,
         class Receiver,
         CUSEND_REQUIRES(is_stream_executor<StreamExecutor>::value),
         CUSEND_REQUIRES(is_receiver_of<Receiver,void>::value),
         CUSEND_REQUIRES(std::is_trivially_copyable<Receiver>::value)
        >
CUSEND_ANNOTATION
event then_execute(const StreamExecutor& ex, event&& e, Receiver receiver)
{
  // get ex's stream
  cudaStream_t stream = detail::stream_of(ex);

  // make stream wait on the predecessor event
  detail::stream_wait_for(stream, e.native_handle());

  // execute the receiver on the executor
  execution::execute(ex, detail::as_trivially_copyable_invocable(receiver));

  // re-record the event on the stream
  e.record_on(stream);

  return std::move(e);
}


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

