#include <cusend/execution/executor/stream_executor.hpp>
#include <cusend/execution/executor/callback_executor.hpp>
#include <cusend/future/detail/is_stream_executor.hpp>

namespace ns = cusend;

void test_is_stream_executor()
{
  static_assert(ns::detail::is_stream_executor<ns::execution::stream_executor>::value, "Error.");
  static_assert(ns::detail::is_stream_executor<ns::execution::callback_executor>::value, "Error.");
}

