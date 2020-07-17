#include <cassert>
#include <cusend/execution/executor/inline_executor.hpp>
#include <cusend/execution/executor/kernel_executor.hpp>
#include <cusend/lazy/combinator/just.hpp>
#include <cusend/lazy/scheduler/as_scheduler.hpp>
#include <cusend/lazy/scheduler/bulk_schedule.hpp>
#include <cusend/lazy/scheduler/device_scheduler.hpp>
#include <cusend/lazy/submit.hpp>


namespace ns = cusend;


#ifndef __host__
#define __host__
#define __device__
#define __managed__
#endif


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


__managed__ bool result0;
__managed__ bool result1;


struct my_receiver
{
  int expected0;
  int expected1;

  __host__ __device__
  void set_value(std::size_t idx) const
  {
    switch(idx)
    {
      case 0:
      {
        result0 = true;
        break;
      }

      case 1:
      {
        result1 = true;
        break;
      }

      default:
      {
        assert(false);
        break;
      }
    }
  }

  __host__ __device__
  void set_value(std::size_t idx, int& value) const
  {
    switch(idx)
    {
      case 0:
      {
        result0 = (expected0 == value);
        break;
      }

      case 1:
      {
        result1 = (expected0 == value);
        break;
      }

      default:
      {
        assert(false);
        break;
      }
    }
  }


  __host__ __device__
  void set_value(std::size_t idx, int& value0, int& value1) const
  {
    switch(idx)
    {
      case 0:
      {
        result0 = (expected0 == value0);
        break;
      }

      case 1:
      {
        result1 = (expected1 == value1);
        break;
      }

      default:
      {
        assert(false);
        break;
      }
    }
  }


  template<class Coord, class... Args>
  __host__ __device__
  void set_value(Coord coord, Args&&... args) const
  {
    return this->set_value(to_index(coord), std::forward<Args>(args)...);
  }


  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


template<class Scheduler>
void test(Scheduler scheduler)
{
  {
    result0 = false;
    result1 = false;

    using coord_type = ns::scheduler_coordinate_t<Scheduler>;
    coord_type shape = to_shape<coord_type>(2);

    auto s0 = ns::just();
    auto s1 = ns::bulk_schedule(scheduler, shape, std::move(s0));

    ns::submit(std::move(s1), my_receiver{13,7});

    cudaStreamSynchronize(0);
    assert(result0);
    assert(result1);
  }

  {
    result0 = false;
    result1 = false;

    using coord_type = ns::scheduler_coordinate_t<Scheduler>;
    coord_type shape = to_shape<coord_type>(2);

    auto s0 = ns::just(13);
    auto s1 = ns::bulk_schedule(scheduler, shape, std::move(s0));

    ns::submit(std::move(s1), my_receiver{13,7});

    cudaStreamSynchronize(0);
    assert(result0);
    assert(result1);
  }

  {
    result0 = false;
    result1 = false;

    using coord_type = ns::scheduler_coordinate_t<Scheduler>;
    coord_type shape = to_shape<coord_type>(2);

    auto s0 = ns::just(13,7);
    auto s1 = ns::bulk_schedule(scheduler, shape, std::move(s0));

    ns::submit(std::move(s1), my_receiver{13,7});

    cudaStreamSynchronize(0);
    assert(result0);
    assert(result1);
  }
}


void test_bulk_schedule()
{
#ifdef __CUDACC__
  // device_scheduler requires CUDA C++
  test(ns::device_scheduler<ns::execution::kernel_executor>());
#endif

  test(ns::as_scheduler(ns::execution::inline_executor{}));
}

