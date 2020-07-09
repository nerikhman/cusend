#include <cusend/execution/executor/stream_executor.hpp>
#include <cusend/lazy/combinator/just.hpp>
#include <cusend/lazy/scheduler/is_device_scheduler.hpp>


namespace ns = cusend;


struct has_schedule_member_function
{
  ns::just_t<> schedule() const;

  bool operator==(const has_schedule_member_function&) const;

  bool operator!=(const has_schedule_member_function&) const;
};


struct has_schedule_free_function
{
  bool operator==(const has_schedule_free_function&) const;

  bool operator!=(const has_schedule_free_function&) const;
};

ns::just_t<> schedule(const has_schedule_free_function&);


template<class Base>
struct has_executor_member_function : Base
{
  ns::execution::stream_executor executor() const;
};


template<class Base>
struct has_get_executor_free_function : Base {};

template<class Base>
ns::execution::stream_executor get_executor(const has_get_executor_free_function<Base>&);


struct not_a_device_scheduler {};

void test_is_device_scheduler()
{
  static_assert(!ns::is_device_scheduler<has_schedule_member_function>::value, "Error.");
  static_assert(!ns::is_device_scheduler<has_schedule_free_function>::value, "Error.");
  static_assert(!ns::is_device_scheduler<not_a_device_scheduler>::value, "Error.");

  static_assert(ns::is_device_scheduler<has_executor_member_function<has_schedule_member_function>>::value, "Error.");
  static_assert(ns::is_device_scheduler<has_executor_member_function<has_schedule_free_function>>::value, "Error.");

  static_assert(ns::is_device_scheduler<has_get_executor_free_function<has_schedule_member_function>>::value, "Error.");
  static_assert(ns::is_device_scheduler<has_get_executor_free_function<has_schedule_free_function>>::value, "Error.");
}

