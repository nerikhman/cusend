#include <cassert>
#include <cudex/sender/set_error.hpp>


#ifndef __CUDACC__
#define __host__
#define __device__
#endif


struct has_set_error_member
{
  __host__ __device__
  bool set_error(int) && noexcept
  {
    return true;
  }
};


struct has_set_error_free_function {};

__host__ __device__
bool set_error(has_set_error_free_function&&, int) noexcept
{
  return true;
}


__host__ __device__
void test()
{
  {
    has_set_error_member r;

    assert(cudex::set_error(std::move(r), 13));
  }

  {
    has_set_error_free_function r;

    assert(cudex::set_error(std::move(r), 13));
  }
}

#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


void test_set_error()
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

