#include <cassert>
#include <cusend/sender/start.hpp>


#ifndef __host__
#define __host__
#define __device__
#endif


struct has_start_member_function
{
  bool started = false;

  __host__ __device__
  void start()
  {
    started = true;
  }
};


struct has_start_free_function
{
  bool started = false;
};

__host__ __device__
void start(has_start_free_function& o)
{
  o.started = true;
}


__host__ __device__
void test()
{
  {
    // test start with member function
    has_start_member_function o;
    cusend::start(o);

    assert(o.started);
  }

  {
    // test sender with free function
    has_start_free_function o;
    cusend::start(o);

    assert(o.started);
  }
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


void test_start()
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

