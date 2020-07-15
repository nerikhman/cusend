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
#include "../../execution/executor/bulk_execute.hpp"
#include "../../execution/executor/executor_index.hpp"
#include "../../execution/executor/executor_shape.hpp"
#include "../../execution/executor/is_device_executor.hpp"
#include "../../lazy/detail/many_receiver_as_trivially_copyable_invocable.hpp"
#include "../../lazy/receiver/is_many_receiver_of.hpp"
#include "event.hpp"
#include "stream_of.hpp"
#include "stream_wait_for.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class DeviceExecutor,
         class ManyReceiver,
         CUSEND_REQUIRES(execution::is_device_executor<DeviceExecutor>::value),
         class Index = execution::executor_index_t<DeviceExecutor>,
         CUSEND_REQUIRES(is_many_receiver_of<ManyReceiver,Index>::value)
        >
CUSEND_ANNOTATION
event then_bulk_execute(const DeviceExecutor& ex, event&& e, ManyReceiver receiver, execution::executor_shape_t<DeviceExecutor> shape)
{
  // get ex's stream
  cudaStream_t stream = detail::stream_of(ex);

  // make stream wait on the predecessor event
  detail::stream_wait_for(stream, e.native_handle());

  // adapt the receiver into an invocable
  many_receiver_as_trivially_copyable_invocable<ManyReceiver> invocable{receiver};

  // bulk_execute on the executor
  execution::bulk_execute(ex, invocable, shape);

  // re-record the event on the stream
  e.record_on(stream);

  return std::move(e);
}


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

