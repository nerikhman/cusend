#include <cusend/execution/executor/inline_executor.hpp>
#include <cusend/execution/executor/is_device_executor.hpp>
#include <cusend/execution/executor/kernel_executor.hpp>


namespace ns = cusend::execution;


void test_is_device_executor()
{
  static_assert(ns::is_device_executor<ns::kernel_executor>::value, "Error.");
  static_assert(!ns::is_device_executor<ns::inline_executor>::value, "Error.");
}

