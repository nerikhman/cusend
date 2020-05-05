#include <cassert>
#include <cudex/sender/set_value.hpp>


#ifndef __CUDACC__
#define __host__
#define __device__
#endif


struct has_set_value_member
{
  __host__ __device__
  bool set_value(int) &&
  {
    return true;
  }
};


struct has_set_value_free_function {};

__host__ __device__
bool set_value(has_set_value_free_function&&, int)
{
  return true;
}


__host__ __device__
void test()
{
  {
    has_set_value_member r;

    assert(cudex::set_value(std::move(r), 13));
  }

  {
    has_set_value_free_function r;

    assert(cudex::set_value(std::move(r), 13));
  }
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


void test_set_value()
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

