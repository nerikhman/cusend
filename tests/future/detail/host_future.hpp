#include <cassert>
#include <utility>
#include <cusend/execution/executor/kernel_executor.hpp>
#include <cusend/future/host_promise.hpp>
#include <cusend/lazy/submit.hpp>

namespace ns = cusend;

template<class Executor>
void test_move_construction(Executor ex)
{
  {
    ns::host_promise<int> p0{};
    ns::detail::host_future<int,Executor> f0 = p0.get_future(ex);
    assert(f0.valid());

    ns::detail::host_future<int,Executor> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
  }

  {
    ns::host_promise<void> p0;
    ns::detail::host_future<void,Executor> f0 = p0.get_future(ex);
    assert(f0.valid());

    ns::detail::host_future<void,Executor> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
  }
}


template<class Executor>
void test_move_assignment(Executor ex)
{
  {
    ns::host_promise<int> p0{};
    ns::detail::host_future<int,Executor> f0 = p0.get_future(ex);
    assert(f0.valid());

    ns::detail::host_future<int,Executor> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
  }


  {
    ns::host_promise<void> p0{};
    ns::detail::host_future<void,Executor> f0 = p0.get_future(ex);
    assert(f0.valid());

    ns::detail::host_future<void,Executor> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
  }
}


template<class Executor, class T>
ns::detail::host_future<T,Executor> make_ready_host_future(const Executor& ex, T value)
{
  ns::host_promise<T> p{};
  std::move(p).set_value(value);
  return p.get_future(ex);
}


template<class Executor>
ns::detail::host_future<void,Executor> make_ready_host_future(const Executor& ex)
{
  ns::host_promise<void> p{};
  std::move(p).set_value();
  return p.get_future(ex);
}


template<class Executor>
void test_get(Executor ex)
{
  {
    ns::detail::host_future<int,Executor> f = make_ready_host_future(ex, 13);
    assert(f.valid());

    assert(13 == std::move(f).get());
    assert(!f.valid());
  }

  {
    ns::detail::host_future<void,Executor> f = make_ready_host_future(ex);
    assert(f.valid());

    std::move(f).get();
    assert(!f.valid());
  }
}


// future.then has several different code paths
// that depend on the type of input and output future
// these tests exercise each case of .then


template<class Executor>
void test_then_void_to_void(Executor ex)
{
  // then void -> void
  auto f1 = make_ready_host_future(ex);
  assert(f1.valid());

  auto f2 = std::move(f1).then(ex, [] __host__ __device__ { return; });

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  std::move(f2).get();
  assert(!f2.valid());
}


template<class Executor>
void test_then_int_to_void(Executor ex)
{
  // then int -> void
  auto f1 = make_ready_host_future(ex, 7);
  assert(f1.valid());

  auto f2 = std::move(f1).then(ex, [] __host__ __device__ (int){return;});

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  std::move(f2).get();
  assert(!f2.valid());
}


template<class Executor>
void test_then_void_to_int(Executor ex)
{
  // then void -> int
  auto f1 = make_ready_host_future(ex);

  auto f2 = std::move(f1).then(ex, [] __host__ __device__ {return 13;});

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  assert(std::move(f2).get() == 13);
  assert(!f2.valid());
}


template<class Executor>
void test_then_int_to_int(Executor ex)
{
  // then int -> int
  auto f1 = make_ready_host_future(ex, 7);

  auto f2 = std::move(f1).then(ex, [] __host__ __device__ (int arg) { return arg + 6; });

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  assert(std::move(f2).get() == 13);
  assert(!f2.valid());
}


template<class Executor>
void test_then_int_to_float(Executor ex)
{
  // then int -> float
  auto f1 = make_ready_host_future(ex, 7);

  auto f2 = std::move(f1).then(ex, [] __host__ __device__ (int arg){return static_cast<float>(arg + 6);});

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  assert(std::move(f2).get() == 13.f);
  assert(!f2.valid());
}


template<class Coord>
__host__ __device__
Coord to_shape(std::size_t size);


template<>
__host__ __device__
std::size_t to_shape<std::size_t>(std::size_t size)
{
  return size;
}

template<>
__host__ __device__
ns::execution::kernel_executor::coordinate_type to_shape<ns::execution::kernel_executor::coordinate_type>(std::size_t size)
{
  return {dim3(1), dim3(size)};
}



__host__ __device__
std::size_t to_index(std::size_t idx)
{
  return idx;
}


__host__ __device__
std::size_t to_index(ns::execution::kernel_executor::coordinate_type coord)
{
  // XXX really ought to generalize this
  return coord.thread.x;
}


template<class Coord>
__host__ __device__
Coord to_coord(std::size_t idx);

template<>
__host__ __device__
std::size_t to_coord<std::size_t>(std::size_t idx)
{
  return idx;
}

template<>
__host__ __device__
ns::execution::kernel_executor::coordinate_type to_coord<ns::execution::kernel_executor::coordinate_type>(std::size_t idx)
{
  return {dim3(0,0,0), dim3(1,0,0)};
}




__managed__ int single_result;


struct receive_void_and_return_void
{
  int result;

  __host__ __device__
  void set_value() && noexcept
  {
    single_result = result;
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


template<class Executor>
void test_then_receiver_of_void_and_return_void(Executor ex)
{
  auto f1 = make_ready_host_future(ex);

  int expected = 13;
  single_result = -1;
  auto f2 = std::move(f1).then(ex, receive_void_and_return_void{expected});

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());
  assert(f2.valid());

  std::move(f2).get();
  assert(!f2.valid());
  assert(expected == single_result);
}


struct receive_void_and_return_int
{
  int result;

  __host__ __device__
  int set_value() && noexcept
  {
    return result;
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


template<class Executor>
void test_then_receiver_of_void_and_return_int(Executor ex)
{
  auto f1 = make_ready_host_future(ex);

  int expected = 13;
  auto f2 = std::move(f1).then(ex, receive_void_and_return_int{expected});

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());
  assert(f2.valid());

  assert(expected == std::move(f2).get());
  assert(!f2.valid());
}


struct receive_int_and_return_void
{
  int result;

  __host__ __device__
  void set_value(int) && noexcept
  {
    single_result = result;
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


template<class Executor>
void test_then_receiver_of_int_and_return_void(Executor ex)
{
  auto f1 = make_ready_host_future(ex, 13);

  int expected = 13;
  single_result = -1;
  auto f2 = std::move(f1).then(ex, receive_int_and_return_void{expected});

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());
  assert(f2.valid());

  std::move(f2).get();
  assert(!f2.valid());
  assert(expected == single_result);
}


struct receive_and_add_one
{
  __host__ __device__
  int set_value(int value) && noexcept
  {
    return value + 1;
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


template<class Executor>
void test_then_receiver_of_int_and_return_int(Executor ex)
{
  int expected = 13;
  auto f1 = make_ready_host_future(ex, expected - 1);

  auto f2 = std::move(f1).then(ex, receive_and_add_one{});

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());
  assert(f2.valid());

  assert(expected == std::move(f2).get());
  assert(!f2.valid());
}


__managed__ bool bulk_result0;
__managed__ bool bulk_result1;


struct many_receiver_of_int
{
  int expected;

  __host__ __device__
  void set_value(int idx, int& value) noexcept
  {
    switch(idx)
    {
      case 0:
      {
        bulk_result0 = (expected == value);
        break;
      }

      case 1:
      {
        bulk_result1 = (expected == value);
        break;
      }

      default:
      {
        assert(0);
        break;
      }
    }
  }

  __host__ __device__
  void set_value(ns::execution::kernel_executor::coordinate_type coord, int& value) noexcept
  {
    set_value(to_index(coord), value);
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


template<class Executor>
void test_bulk_then_receiver_of_int(Executor ex)
{
  int expected = 7;
  auto f1 = make_ready_host_future(ex, std::move(expected));

  bulk_result0 = false;
  bulk_result1 = false;

  using coord_type = ns::execution::executor_coordinate_t<Executor>;
  auto shape = to_shape<coord_type>(2);

  auto f2 = std::move(f1).bulk_then(ex, many_receiver_of_int{expected}, shape);

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  assert(expected == std::move(f2).get());
  assert(bulk_result0);
  assert(bulk_result1);
}


struct many_receiver_of_void
{
  __host__ __device__
  void set_value(int idx) noexcept
  {
    switch(idx)
    {
      case 0:
      {
        bulk_result0 = true;
        break;
      }

      case 1:
      {
        bulk_result1 = true;
        break;
      }

      default:
      {
        assert(0);
        break;
      }
    }
  }

  __host__ __device__
  void set_value(ns::execution::kernel_executor::coordinate_type coord) noexcept
  {
    set_value(to_index(coord));
  }

  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


template<class Executor>
void test_bulk_then_receiver_of_void(Executor ex)
{
  auto f1 = make_ready_host_future(ex);

  bulk_result0 = false;
  bulk_result1 = false;

  using coord_type = ns::execution::executor_coordinate_t<Executor>;
  auto shape = to_shape<coord_type>(2);

  auto f2 = std::move(f1).bulk_then(ex, many_receiver_of_void{}, shape);

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());

  assert(bulk_result0);
  assert(bulk_result1);
}


template<class Executor>
void test_bulk_then_void_to_void(Executor ex)
{
  auto f1 = make_ready_host_future(ex);

  bulk_result0 = false;
  bulk_result1 = false;

  using coord_type = ns::execution::executor_coordinate_t<Executor>;
  coord_type shape = to_shape<coord_type>(2);

  auto f2 = std::move(f1).bulk_then(ex, [] __host__ __device__ (coord_type coord)
  {
    switch(to_index(coord))
    {
      case 0:
      {
        bulk_result0 = true;
        break;
      }

      case 1:
      {
        bulk_result1 = true;
        break;
      }

      default:
      {
        assert(false);
        break;
      }
    }
  }, shape);

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());
  assert(bulk_result0);
  assert(bulk_result1);
}


template<class Executor>
void test_bulk_then_int_to_int(Executor ex)
{
  int expected = 13;
  auto f1 = make_ready_host_future(ex, std::move(expected));

  bulk_result0 = false;
  bulk_result1 = false;

  using coord_type = ns::execution::executor_coordinate_t<Executor>;
  coord_type shape = to_shape<coord_type>(2);

  auto f2 = std::move(f1).bulk_then(ex, [] __host__ __device__ (coord_type coord, int& value)
  {
    switch(to_index(coord))
    {
      case 0:
      {
        bulk_result0 = (13 == value);
        break;
      }

      case 1:
      {
        bulk_result1 = (13 == value);
        break;
      }

      default:
      {
        assert(false);
        break;
      }
    }
  }, shape);

  assert(!f1.valid());
  assert(f2.valid());
  f2.wait();

  assert(f2.is_ready());
  assert(expected == std::move(f2).get());
  assert(bulk_result0);
  assert(bulk_result1);
}


template<class Executor>
void test_bulk(Executor ex)
{
  {
    // test void future

    bulk_result0 = false;
    bulk_result1 = false;

    using coord_type = ns::execution::executor_coordinate_t<Executor>;
    coord_type shape = to_shape<coord_type>(2);

    ns::submit(make_ready_host_future(ex).bulk(shape), many_receiver_of_void{});
    assert(cudaSuccess == cudaDeviceSynchronize());

    assert(bulk_result0);
    assert(bulk_result1);
  }

  {
    // test int future

    bulk_result0 = false;
    bulk_result1 = false;

    using coord_type = ns::execution::executor_coordinate_t<Executor>;
    coord_type shape = to_shape<coord_type>(2);

    ns::submit(make_ready_host_future(ex, 13).bulk(shape), many_receiver_of_int{13});
    assert(cudaSuccess == cudaDeviceSynchronize());

    assert(bulk_result0);
    assert(bulk_result1);
  }
}


template<class Executor>
void test(Executor ex)
{
  test_move_construction(ex);
  test_move_assignment(ex);
  test_get(ex);

  test_then_void_to_void(ex);
  test_then_void_to_int(ex);
  test_then_int_to_void(ex);
  test_then_int_to_int(ex);
  test_then_int_to_float(ex);

  test_then_receiver_of_void_and_return_void(ex);
  test_then_receiver_of_void_and_return_int(ex);
  test_then_receiver_of_int_and_return_void(ex);
  test_then_receiver_of_int_and_return_int(ex);

  test_bulk_then_receiver_of_void(ex);
  test_bulk_then_receiver_of_int(ex);

  test_bulk_then_void_to_void(ex);
  test_bulk_then_int_to_int(ex);

  test_bulk(ex);
}


void test_host_future()
{
#ifdef __CUDACC__
  // kernel_executor launches kernels, so only test it with a CUDA compiler
  test(ns::execution::kernel_executor{});
#endif

  test(ns::execution::callback_executor{});
}

