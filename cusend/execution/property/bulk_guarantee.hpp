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

#include "../detail/prologue.hpp"

#include "../executor/is_executor.hpp"
#include "detail/basic_executor_property.hpp"
#include "detail/has_static_query_member_function.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


struct bulk_guarantee_t :
  detail::basic_executor_property<bulk_guarantee_t, false, false, bulk_guarantee_t>
{
  CUDEX_ANNOTATION
  friend constexpr bool operator==(const bulk_guarantee_t& a, const bulk_guarantee_t& b)
  {
    return a.which_ == b.which_;
  }

  CUDEX_ANNOTATION
  friend constexpr bool operator!=(const bulk_guarantee_t& a, const bulk_guarantee_t& b)
  {
    return !(a == b);
  }

  CUDEX_ANNOTATION
  constexpr bulk_guarantee_t()
    : which_{0}
  {}

  struct sequenced_t :
    detail::basic_executor_property<sequenced_t, true, true>
  {
    CUDEX_ANNOTATION
    static constexpr sequenced_t value()
    {
      return sequenced_t{};
    }
  };

  static constexpr sequenced_t sequenced{};


  CUDEX_ANNOTATION
  constexpr bulk_guarantee_t(const sequenced_t&)
    : which_{1}
  {}

  struct parallel_t :
    detail::basic_executor_property<parallel_t, true, true>
  {
    CUDEX_ANNOTATION
    static constexpr parallel_t value()
    {
      return parallel_t{};
    }
  };

  static constexpr parallel_t parallel{};

  CUDEX_ANNOTATION
  constexpr bulk_guarantee_t(const parallel_t&)
    : which_{2}
  {}


  struct unsequenced_t :
    detail::basic_executor_property<unsequenced_t, true, true>
  {
    CUDEX_ANNOTATION
    static constexpr unsequenced_t value()
    {
      return unsequenced_t{};
    }
  };

  static constexpr unsequenced_t unsequenced{};

  CUDEX_ANNOTATION
  constexpr bulk_guarantee_t(const unsequenced_t&)
    : which_{3}
  {}


  struct concurrent_t :
    detail::basic_executor_property<concurrent_t, true, true>
  {
    CUDEX_ANNOTATION
    static constexpr concurrent_t value()
    {
      return concurrent_t{};
    }
  };

  static constexpr concurrent_t concurrent{};

  CUDEX_ANNOTATION
  constexpr bulk_guarantee_t(const concurrent_t&)
    : which_{4}
  {}


  // By default, executors are unsequenced if bulk_guarantee_t cannot
  // be statically-queried through a member
  template<class Executor,
           CUDEX_REQUIRES(
             !detail::has_static_query_member_function<Executor, bulk_guarantee_t>::value
           )>
  CUDEX_ANNOTATION
  constexpr static unsequenced_t static_query()
  {
    return unsequenced_t{};
  }


  template<class Executor,
           CUDEX_REQUIRES(
             detail::has_static_query_member_function<Executor, bulk_guarantee_t>::value
           )>
  CUDEX_ANNOTATION
  static constexpr auto static_query()
    -> decltype(Executor::query(std::declval<bulk_guarantee_t>()))
  {
    return Executor::query(bulk_guarantee_t{});
  }


  template<class OuterGuarantee, class InnerGuarantee>
  class scoped_t
  {
    public:
      static constexpr bool is_requirable = true;
      static constexpr bool is_preferable = true;

      template<class Executor,
               CUDEX_REQUIRES(
                 detail::has_static_query_member_function<Executor, scoped_t>::value
               )>
      CUDEX_ANNOTATION
      static constexpr auto static_query()
        -> decltype(Executor::query(std::declval<scoped_t>()))
      {
        return Executor::query(scoped_t{});
      }

      scoped_t() = default;

      scoped_t(const scoped_t&) = default;

      CUDEX_ANNOTATION
      constexpr scoped_t(const OuterGuarantee& outer, const InnerGuarantee& inner)
        : outer_{outer}, inner_{inner}
      {}

      CUDEX_ANNOTATION
      constexpr OuterGuarantee outer() const
      {
        return outer_;
      }

      CUDEX_ANNOTATION
      constexpr InnerGuarantee inner() const
      {
        return inner_;
      }

      CUDEX_ANNOTATION
      constexpr scoped_t value() const
      {
        return *this;
      }

      CUDEX_ANNOTATION
      friend constexpr bool operator==(const scoped_t& a, const scoped_t& b)
      {
        return (a.outer().value() == b.outer().value()) && (a.inner().value() == b.inner().value());
      }

      CUDEX_ANNOTATION
      friend constexpr bool operator!=(const scoped_t& a, const scoped_t& b)
      {
        return !(a == b);
      }

    private:
      OuterGuarantee outer_;
      InnerGuarantee inner_;
  }; // end scoped_t


  template<class OuterGuarantee, class InnerGuarantee>
  CUDEX_ANNOTATION
  constexpr static scoped_t<OuterGuarantee,InnerGuarantee> scoped(OuterGuarantee, InnerGuarantee)
  {
    return {};
  }


  private:
    int which_;
};


#ifndef __CUDA_ARCH__
static constexpr bulk_guarantee_t bulk_guarantee{};
#else
const __device__ bulk_guarantee_t bulk_guarantee{};
#endif


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

