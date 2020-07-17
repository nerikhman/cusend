#include <cassert>
#include <cusend/execution/executor/kernel_executor.hpp>
#include <cusend/execution/property/stream.hpp>
#include <cusend/property/query.hpp>


namespace ns = cusend;


void test_stream_property()
{
  cudaStream_t expected{};
  assert(cudaSuccess == cudaStreamCreate(&expected));

  ns::execution::kernel_executor ex{expected};
  assert(expected == cusend::query(ex, ns::execution::stream));

  assert(cudaSuccess == cudaStreamDestroy(expected));
}


void test_query()
{
  test_stream_property();
}

