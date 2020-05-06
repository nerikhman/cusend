#include <cassert>
#include <cusend/memory/allocator/allocation_deleter.hpp>
#include <cusend/memory/allocator/allocator.hpp>
#include <limits>

namespace ns = cusend::memory;


#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __managed__
#define __managed__
#endif


__host__ __device__
void test_constructors()
{
  ns::allocator<int> expected;

  ns::allocation_deleter<ns::allocator<int>> allocator_constructed{expected};
  assert(expected == allocator_constructed.allocator());

  ns::allocation_deleter<ns::allocator<int>>  copy_constructed{allocator_constructed};
  assert(expected == copy_constructed.allocator());

  ns::allocation_deleter<ns::allocator<void>> converting_copy_constructed{allocator_constructed};
  assert(expected == converting_copy_constructed.allocator());
}



struct set_on_delete
{
  bool& deleted;

  __host__ __device__
  ~set_on_delete()
  {
    deleted = true;
  }
};


__managed__ bool deleted;


__host__ __device__
void test_call_operator()
{
  ns::allocator<set_on_delete> alloc;
  ns::allocation_deleter<ns::allocator<set_on_delete>> deleter{alloc};

  set_on_delete* ptr = alloc.allocate(1);

  deleted = false;
  new(ptr) set_on_delete{deleted};

  deleter(ptr);

  assert(deleted = true);
}


template<class T>
class stateful_allocator : public ns::allocator<T>
{
  public:
    stateful_allocator(const stateful_allocator&) = default;

    __host__ __device__
    stateful_allocator(int state) : state_(state) {}
  
    __host__ __device__
    bool operator==(const stateful_allocator& other) const
    {
      return state_ == other.state_;
    }
  
  private:
    int state_;
};


__host__ __device__
void test_swap()
{
  stateful_allocator<set_on_delete> alloc0{0}, alloc1{1};

  ns::allocation_deleter<stateful_allocator<set_on_delete>> deleter_a(alloc0);
  ns::allocation_deleter<stateful_allocator<set_on_delete>> deleter_b(alloc1);

  deleter_a.swap(deleter_b);

  assert(alloc1 == deleter_a.allocator());
  assert(alloc0 == deleter_b.allocator());
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


void test_allocation_deleter()
{
  test_call_operator();
  test_constructors();
  test_swap();

#ifdef __CUDACC__
  device_invoke<<<1,1>>>([] __device__ ()
  {
    test_call_operator();
    test_constructors();
    test_swap();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

