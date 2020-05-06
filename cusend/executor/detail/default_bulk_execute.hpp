#pragma once

#include "../../detail/prologue.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>
#include "../../detail/type_traits/is_detected.hpp"
#include "../../detail/type_traits/is_invocable.hpp"
#include "../../detail/type_traits/remove_cvref.hpp"
#include "../../property/bulk_guarantee.hpp"
#include "../../property/detail/static_query.hpp"
#include "../execute.hpp"
#include "../is_executor.hpp"


CUSEND_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class E>
using has_unsequenced_bulk_guarantee = is_detected_exact<
  bulk_guarantee_t::unsequenced_t,
  static_query_t,
  bulk_guarantee_t,
  remove_cvref_t<E>
>;


template<class Function>
struct invoke_with_index
{
  mutable Function f;
  std::size_t i;

  CUSEND_ANNOTATION
  void operator()() const
  {
    f(i);
  }
};


CUSEND_EXEC_CHECK_DISABLE
template<class Executor, class Function,
         CUSEND_REQUIRES(is_executor<Executor>::value),
         CUSEND_REQUIRES(has_unsequenced_bulk_guarantee<Executor>::value),
         CUSEND_REQUIRES(is_invocable<remove_cvref_t<Function>,std::size_t>::value),
         CUSEND_REQUIRES(std::is_copy_constructible<remove_cvref_t<Function>>::value)
        >
CUSEND_ANNOTATION
void default_bulk_execute(const Executor& ex, Function&& f, std::size_t shape)
{
  for(std::size_t i = 0; i < shape; ++i)
  {
    CUSEND_NAMESPACE::execute(ex, invoke_with_index<remove_cvref_t<Function>>{f, i});
  }
}


template<class Function>
struct invoke_reference_with_index
{
  Function& f;
  std::size_t i;

  CUSEND_ANNOTATION
  void operator()() const
  {
    f(i);
  }
};


CUSEND_EXEC_CHECK_DISABLE
template<class Executor, class Function,
         CUSEND_REQUIRES(is_executor<Executor>::value),
         CUSEND_REQUIRES(has_unsequenced_bulk_guarantee<Executor>::value),
         CUSEND_REQUIRES(is_invocable<remove_cvref_t<Function>&,std::size_t>::value),
         CUSEND_REQUIRES(!std::is_copy_constructible<remove_cvref_t<Function>>::value)
        >
CUSEND_ANNOTATION
void default_bulk_execute(const Executor& ex, Function&& f, std::size_t shape)
{
  for(std::size_t i = 0; i < shape; ++i)
  {
    CUSEND_NAMESPACE::execute(ex, invoke_reference_with_index<remove_cvref_t<Function>>{f, i});
  }
}


template<class E, class F, class S>
using default_bulk_execute_t = decltype(detail::default_bulk_execute(std::declval<E>(), std::declval<F>(), std::declval<S>()));


template<class E, class F, class S>
using can_default_bulk_execute = is_detected<default_bulk_execute_t, E, F, S>;


} // end detail


CUSEND_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

