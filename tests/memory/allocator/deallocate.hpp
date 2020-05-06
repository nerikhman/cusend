#include <cassert>
#include <cusend/memory/allocator/allocator.hpp>
#include <cusend/memory/allocator/deallocate.hpp>
#include <limits>

namespace ns = cusend::memory;


#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


__host__ __device__
void test()
{
  ns::allocator<int> a;

  int* ptr = a.allocate(1);

  int expected = 13;
  *ptr = expected;
  int result = *ptr;

  assert(expected == result);

  ns::deallocate(a, ptr, 1);
}


void test_deallocate()
{
  test();

#ifdef __CUDACC__
  device_invoke<<<1,1>>>([] __device__ ()
  {
    test();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

