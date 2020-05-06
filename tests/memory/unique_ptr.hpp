#include <cassert>
#include <cusend/memory/unique_ptr.hpp>
#include <cusend/memory/allocator/allocator_delete.hpp>
#include <cusend/memory/allocator/allocator_new.hpp>

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

struct base {};
struct derived : base {};


__managed__ bool deleted;


template<class T>
struct my_deleter
{
  bool* deleted;

  using pointer = T*;

  __host__ __device__
  void operator()(T* ptr)
  {
    ns::allocator<T> alloc;
    ns::allocator_delete(alloc, ptr);
    *deleted = true;
  }

  __host__ __device__
  my_deleter(bool* d) : deleted(d) {}

  __host__ __device__
  my_deleter() : my_deleter(nullptr) {}

  template<class U>
  __host__ __device__
  my_deleter(const my_deleter<U>& other) : my_deleter(other.deleted) {}
};


__host__ __device__
void test_constructors()
{
  {
    // with default_delete

    ns::allocator<int> alloc;
    int expected = 13;

    // pointer ctor
    ns::unique_ptr<int> p0(ns::allocator_new<int>(alloc, expected));
    assert(expected == *p0);

    // null ctor
    ns::unique_ptr<int> p1;
    assert(!p1);

    // move ctor
    ns::unique_ptr<int> p2 = std::move(p0);
    assert(expected == *p2);
    assert(!p0);

    // converting ctor
    ns::unique_ptr<derived> pd;
    ns::unique_ptr<base> pb = std::move(pd);
  }

  {
    // with my_deleter

    ns::allocator<int> alloc;
    int expected = 13;

    deleted = false;
    {
      // pointer ctor
      ns::unique_ptr<int, my_deleter<int>> p0(ns::allocator_new<int>(alloc, expected), my_deleter<int>{&deleted});
      assert(expected == *p0);

      // move ctor
      ns::unique_ptr<int, my_deleter<int>> p2 = std::move(p0);
      assert(expected == *p2);
      assert(!p0);
    }
    assert(deleted);

    // converting ctor
    ns::unique_ptr<derived, my_deleter<derived>> pd(nullptr, my_deleter<derived>{&deleted});
    ns::unique_ptr<base, my_deleter<base>> pb = std::move(pd);
  }
}


struct set_on_delete
{
  bool& deleted;

  __host__ __device__
  set_on_delete(bool& d) : deleted(d) {}

  __host__ __device__
  ~set_on_delete()
  {
    deleted = true;
  }
};


__host__ __device__
void test_destructor()
{
  {
    // with default_delete
    deleted = false;
    ns::allocator<set_on_delete> alloc;

    {
      ns::unique_ptr<set_on_delete> p0(ns::allocator_new<set_on_delete>(alloc, deleted));
    }

    assert(deleted);
  }
}


__host__ __device__
void test_move_assignment()
{
  int expected = 13;

  ns::unique_ptr<int> p0 = ns::make_unique<int>(expected);
  ns::unique_ptr<int> p1;
  p1 = std::move(p0);

  assert(expected == *p1);
  assert(!p0);
}


__host__ __device__
void test_get()
{
  ns::allocator<int> alloc;
  int* expected = ns::allocator_new<int>(alloc);

  ns::unique_ptr<int> p0{expected};
  assert(expected == p0.get());
}


__host__ __device__
void test_release()
{
  ns::allocator<int> alloc;
  int* expected = ns::allocator_new<int>(alloc);

  ns::unique_ptr<int> p0{expected};
  int* result = p0.release();
  assert(expected == result);
  assert(!p0);

  ns::allocator_delete(alloc, expected);
}


__host__ __device__
void test_reset()
{
  ns::allocator<int> alloc;

  {
    // reset null ptr

    int* ptr = ns::allocator_new<int>(alloc);
    ns::unique_ptr<int> p;
    p.reset(ptr);

    assert(ptr == p.get());
  }

  {
    // reset non-null ptr

    ns::unique_ptr<int> p{ns::allocator_new<int>(alloc)};

    int* expected = ns::allocator_new<int>(alloc);
    p.reset(expected);

    assert(expected == p.get());
  }
}


template<class T>
__host__ __device__
const T& cref(const T& ref)
{
  return ref;
}


__host__ __device__
void test_get_deleter()
{
  ns::unique_ptr<int> p;

  p.get_deleter();
  cref(p).get_deleter();
}


__host__ __device__
void test_operator_star()
{
  int expected = 13;

  ns::unique_ptr<int> p = ns::make_unique<int>(13);
  assert(expected == *p);
  assert(expected == *cref(p));
}


__host__ __device__
void test_operator_bool()
{
  ns::unique_ptr<int> null;
  assert(!null);

  ns::unique_ptr<int> p = ns::make_unique<int>(13);
  assert(bool(p));
}


__host__ __device__
void test_swap()
{
  int expected1 = 13;
  int expected2 = 7;

  ns::unique_ptr<int> p1 = ns::make_unique<int>(expected1);
  ns::unique_ptr<int> p2 = ns::make_unique<int>(expected2);

  p1.swap(p2);

  assert(expected1 == *p2);
  assert(expected2 == *p1);
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


void test_unique_ptr()
{
  test_constructors();
  test_destructor();
  test_move_assignment();
  test_get();
  test_release();
  test_reset();
  test_get_deleter();
  test_operator_star();
  test_operator_bool();

#ifdef __CUDACC__
  device_invoke<<<1,1>>>([] __device__ ()
  {
    test_constructors();
    test_destructor();
    test_move_assignment();
    test_get();
    test_release();
    test_reset();
    test_get_deleter();
    test_operator_star();
    test_operator_bool();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

