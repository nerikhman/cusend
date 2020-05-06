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

#include "prologue.hpp"

#include <cuda_runtime_api.h>
#include <utility>
#include "kernel_operation.hpp"

CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Function>
class kernel_sender
{
  private:
    static_assert(std::is_trivially_copyable<Function>::value, "Function must be trivially copyable.");

  public:
    CUSEND_ANNOTATION
    kernel_sender(Function kernel, dim3 grid_dim, dim3 block_dim, std::size_t shared_memory_size, cudaStream_t stream, int device) noexcept
      : kernel_(kernel),
        grid_dim_(grid_dim),
        block_dim_(block_dim),
        shared_memory_size_(shared_memory_size),
        stream_(stream),
        device_(device)
    {}

    CUSEND_ANNOTATION
    recorded_kernel_operation<Function> connect(cudaEvent_t recording_event) && noexcept
    {
      return {kernel_, grid_dim_, block_dim_, shared_memory_size_, stream_, device_, recording_event};
    }

    // what should we actually do in general?
    // run the receiver in a host callback?
    // enqueue the receiver on a thread pool?
    //template<class Receiver,
    //         CUSEND_REQUIRES(is_receiver<Receiver>::value)
    //        >
    //void connect(Receiver&& r);

  private:
    Function kernel_;
    dim3 grid_dim_;
    dim3 block_dim_;
    std::size_t shared_memory_size_;
    cudaStream_t stream_;
    int device_;
};


template<class Function>
CUSEND_ANNOTATION
kernel_sender<Function> make_kernel_sender(Function f, dim3 grid_dim, dim3 block_dim, std::size_t shared_memory_size, cudaStream_t stream, int device) noexcept
{
  return {f, grid_dim, block_dim, shared_memory_size, stream, device};
}


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

