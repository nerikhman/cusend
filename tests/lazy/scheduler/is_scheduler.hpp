#include <cusend/execution/executor/inline_executor.hpp>
#include <cusend/lazy/just.hpp>
#include <cusend/lazy/scheduler/is_scheduler.hpp>


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


struct not_a_scheduler {};

void test_is_scheduler()
{
  static_assert(ns::is_scheduler<has_schedule_member_function>::value, "Error.");
  static_assert(ns::is_scheduler<has_schedule_free_function>::value, "Error.");
  static_assert(ns::is_scheduler<ns::execution::inline_executor>::value, "Error.");
  static_assert(!ns::is_scheduler<not_a_scheduler>::value, "Error.");
}

