#include <cassert>
#include <tuple>
#include <utility>
#include <cusend/execution/executor/executor_index.hpp>
#include <cusend/execution/executor/inline_executor.hpp>
#include <cusend/execution/executor/stream_executor.hpp>


namespace ns = cusend::execution;


#ifndef __host__
#define __host__
#define __device__
#endif


struct empty {};


struct has_nested_shape_type
{
  using shape_type = std::pair<int,int>;
};


struct has_nested_index_type
{
  using index_type = std::tuple<int,int,int>;
};


__host__ __device__
void test()
{
  static_assert(std::is_same<std::size_t, ns::executor_index_t<empty>>::value, "Error.");
  static_assert(std::is_same<std::pair<int,int>, ns::executor_index_t<has_nested_shape_type>>::value, "Error.");
  static_assert(std::is_same<std::tuple<int,int,int>, ns::executor_index_t<has_nested_index_type>>::value, "Error.");
  static_assert(std::is_same<std::size_t, ns::executor_index_t<ns::inline_executor>>::value, "Error.");
  static_assert(std::is_same<std::size_t, ns::executor_index_t<ns::stream_executor>>::value, "Error.");
}


void test_executor_index()
{
  test();
}

