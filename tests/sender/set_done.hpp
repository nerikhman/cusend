#include <cassert>
#include <cusend/sender/set_done.hpp>


#ifndef __CUDACC__
#define __host__
#define __device__
#endif


struct has_set_done_member
{
  __host__ __device__
  bool set_done() && noexcept
  {
    return true;
  }
};


struct has_set_done_free_function {};

__host__ __device__
bool set_done(has_set_done_free_function&&) noexcept
{
  return true;
}


__host__ __device__
void test()
{
  {
    has_set_done_member r;

    assert(cusend::set_done(std::move(r)));
  }

  {
    has_set_done_free_function r;

    assert(cusend::set_done(std::move(r)));
  }
}

#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


void test_set_done()
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

