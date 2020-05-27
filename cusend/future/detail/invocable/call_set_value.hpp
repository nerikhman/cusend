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
#include "../../../sender/set_error.hpp"
#include "../../../sender/set_value.hpp"

CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Receiver>
struct call_set_value
{
  Receiver r;

  template<class... Args>
  CUSEND_ANNOTATION
  void operator()(Args&&... args)
  {
#ifdef __CUDA_ARCH__
    CUSEND_NAMESPACE::set_value(std::move(r), std::forward<Args>(args)...);
#else
    try
    {
      CUSEND_NAMESPACE::set_value(std::move(r), std::forward<Args>(args)...);
    }
    catch(...)
    {
      CUSEND_NAMESPACE::set_error(std::move(r), std::current_exception());
    }
#endif
  }
};


template<class Receiver,
         CUSEND_REQUIRES(is_receiver<Receiver>::value),
         CUSEND_REQUIRES(std::is_trivially_copy_constructible<Receiver>::value)
        >
CUSEND_ANNOTATION
call_set_value<remove_cvref_t<Receiver>> make_call_set_value(Receiver r)
{
  return {r};
}


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../../detail/epilogue.hpp"

