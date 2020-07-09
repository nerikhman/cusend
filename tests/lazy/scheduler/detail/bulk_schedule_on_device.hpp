#include <cassert>
#include <cusend/lazy/combinator/just.hpp>
#include <cusend/lazy/scheduler/detail/bulk_schedule_on_device.hpp>
#include <cusend/lazy/scheduler/device_scheduler.hpp>
#include <cusend/lazy/submit.hpp>


namespace ns = cusend;


#ifndef __host__
#define __host__
#define __device__
#define __managed__
#endif


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

    printf("my_receiver::set_value: idx %d received %d\n", (int)idx, value);
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

    printf("my_receiver::set_value: idx %d received (%d, %d)\n", (int)idx, value0, value1);
  }


  template<class E>
  __host__ __device__
  void set_error(E&&) && noexcept {}

  __host__ __device__
  void set_done() && noexcept {}
};


void test_bulk_schedule_on_device()
{
#ifdef __CUDACC__
  // device_scheduler requires CUDA C++
  ns::device_scheduler<ns::execution::stream_executor> scheduler;

  {
    result0 = false;
    result1 = false;

    auto s0 = ns::just();
    auto s1 = ns::detail::bulk_schedule_on_device(scheduler, 2, std::move(s0));

    ns::submit(std::move(s1), my_receiver{13,7});

    cudaStreamSynchronize(scheduler.executor().stream());
    assert(result0);
    assert(result1);
  }

  {
    result0 = false;
    result1 = false;

    auto s0 = ns::just(13);
    auto s1 = ns::detail::bulk_schedule_on_device(scheduler, 2, std::move(s0));

    ns::submit(std::move(s1), my_receiver{13,7});

    cudaStreamSynchronize(scheduler.executor().stream());
    assert(result0);
    assert(result1);
  }

  {
    result0 = false;
    result1 = false;

    auto s0 = ns::just(13,7);
    auto s1 = ns::detail::bulk_schedule_on_device(scheduler, 2, std::move(s0));

    ns::submit(std::move(s1), my_receiver{13,7});

    cudaStreamSynchronize(scheduler.executor().stream());
    assert(result0);
    assert(result1);
  }
#endif // __CUDACC__
}

