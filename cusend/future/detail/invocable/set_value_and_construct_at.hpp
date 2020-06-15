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

#include <exception>
#include <type_traits>
#include <utility>
#include "../../../lazy/receiver/set_error.hpp"
#include "../../../lazy/receiver/set_value.hpp"
#include "../../../memory/allocator/construct_at.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Receiver, class ResultPointer>
struct set_value_and_construct_at
{
  Receiver r;
  ResultPointer result_ptr;

  CUSEND_ANNOTATION
  void operator()()
  {
#ifdef __CUDA_ARCH__
    memory::construct_at(result_ptr, CUSEND_NAMESPACE::set_value(std::move(r)));
#else
    try
    {
      memory::construct_at(result_ptr, CUSEND_NAMESPACE::set_value(std::move(r)));
    }
    catch(...)
    {
      CUSEND_NAMESPACE::set_error(std::move(r), std::current_exception());
    }
#endif
  }
};

template<class Receiver, class ResultPointer,
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<Receiver>::value),
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<ResultPointer>::value)
        >
CUSEND_ANNOTATION
set_value_and_construct_at<Receiver,ResultPointer> make_set_value_and_construct_at(Receiver r, ResultPointer result_ptr)
{
  return {r, result_ptr};
}


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../../detail/epilogue.hpp"

