#include <cassert>
#include <cusend/memory/allocator/managed_allocator.hpp>
#include <limits>

namespace ns = cusend::memory;


void test_allocate()
{
  ns::managed_allocator<int> a;

  int* ptr = a.allocate(1);

  cudaPointerAttributes attr{};
  assert(cudaPointerGetAttributes(&attr, ptr) == cudaSuccess);
  assert(attr.type == cudaMemoryTypeManaged);
  assert(attr.device == a.device());
  assert(attr.devicePointer == ptr);
  assert(attr.hostPointer == ptr);

  int expected = 13;
  *ptr = expected;
  int result = *ptr;

  assert(expected == result);

  a.deallocate(ptr, 1);
}


void test_comparison()
{
  ns::managed_allocator<int> a0{0};
  ns::managed_allocator<int> a1{1};

  // same allocator compares same
  assert(a0 == a0);
  assert(!(a0 != a0));

  // allocators pointing to same device compare same
  ns::managed_allocator<int> other_a0{0};
  assert(a0 == other_a0);
  assert(!(a0 != other_a0));

  // allocators pointing to different devices compare different
  assert(!(a0 == a1));
  assert(a0 != a1);
}


void test_copy_construction()
{
  ns::managed_allocator<int> a;
  ns::managed_allocator<int> copy = a;

  assert(a == copy);
  assert(!(a != copy));
}


void test_converting_copy_construction()
{
  ns::managed_allocator<int> a;
  ns::managed_allocator<float> copy = a;

  // silence unused variable warnings
  (void)copy;
}


void test_device()
{
  ns::managed_allocator<int> a;
  assert(a.device() == 0);

  ns::managed_allocator<int> a1{1};
  assert(a1.device() == 1);
}


void test_throw_on_failure()
{
  ns::managed_allocator<int> a;

  try
  {
    std::size_t n = std::numeric_limits<std::size_t>::max();
    a.allocate(n);
  }
  catch(...)
  {
    return;
  }

  assert(0);
}


void test_managed_allocator()
{
  test_allocate();
  test_comparison();
  test_copy_construction();
  test_converting_copy_construction();
  test_device();
  test_throw_on_failure();
}

