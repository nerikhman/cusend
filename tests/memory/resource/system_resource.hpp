#include <cassert>
#include <cusend/memory/resource/managed_resource.hpp>
#include <cusend/memory/resource/system_resource.hpp>
#include <limits>
#include <utility>

namespace ns = cusend::memory;


void test_allocate()
{
  using system_resource = ns::system_resource<ns::managed_resource>;

  system_resource r{0};

  int* ptr = static_cast<int*>(r.allocate(sizeof(int)));

  cudaPointerAttributes attr{};
  assert(cudaPointerGetAttributes(&attr, ptr) == cudaSuccess);
  assert(attr.type == cudaMemoryTypeManaged);
  assert(attr.devicePointer == ptr);
  assert(attr.hostPointer == ptr);

  int expected = 13;
  *ptr = expected;
  int result = *ptr;

  assert(expected == result);

  r.deallocate(ptr, sizeof(int));
}


void test_comparison()
{
  using system_resource = ns::system_resource<ns::managed_resource>;

  system_resource r0{0};
  system_resource r1{1};

  // same resource compares same
  assert(r0.is_equal(r0));
  assert(r0 == r0);
  assert(!(r0 != r0));

  // different resources compare different
  assert(!r0.is_equal(r1));
  assert(r0 != r1);

  // resources pointing to same device compare same
  system_resource other_r0{0};
  assert(r0.is_equal(other_r0));
  assert(r0 == other_r0);
  assert(!(r0 != other_r0));
}


void test_copy_construction()
{
  using system_resource = ns::system_resource<ns::managed_resource>;

  system_resource r0{0};
  system_resource copy = r0;

  assert(r0 == copy);
}


void test_device()
{
  using system_resource = ns::system_resource<ns::managed_resource>;

  system_resource r;
  assert(r.device() == 0);

  system_resource r1{1};
  assert(r1.device() == 1);
}


void test_throw_on_failure()
{
  using system_resource = ns::system_resource<ns::managed_resource>;

  system_resource r;

  try
  {
    std::size_t num_bytes = std::numeric_limits<std::size_t>::max();
    r.allocate(num_bytes);
  }
  catch(...)
  {
    return;
  }

  assert(0);
}


void test_system_resource()
{
  test_allocate();
  test_comparison();
  test_copy_construction();
  test_device();
  test_throw_on_failure();
}

