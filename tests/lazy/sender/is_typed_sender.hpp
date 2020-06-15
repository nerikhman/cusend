#include <cassert>
#include <cusend/sender/is_typed_sender.hpp>
#include <cusend/sender/set_value.hpp>


struct not_a_sender {};


struct untyped_sender
{
  template<class R>
  void submit(R&& r) &&
  {
    cusend::set_value(r);
  }
};


struct sender_with_value_types : untyped_sender
{
  template<template<class...> class Tuple, template<class...> class Variant>
  using value_types = Variant<Tuple<>>;
};


struct sender_with_value_types_and_error_types : sender_with_value_types
{
  template<template<class...> class Variant>
  using error_types = Variant<>;
};


struct sender_with_all_types : sender_with_value_types_and_error_types
{
  constexpr static bool sends_done = false;
};


struct sender_that_specializes_sender_traits : untyped_sender {};

namespace cusend
{

template<>
struct sender_traits<::sender_that_specializes_sender_traits> : public sender_traits<sender_with_all_types> {};

}


void test_is_typed_sender()
{
  using namespace cusend;

  // test not a sender without any traits
  static_assert(!is_typed_sender<not_a_sender>::value, "Expected false.");

  // test an untyped sender without any traits
  static_assert(!is_typed_sender<untyped_sender>::value, "Expected false.");

  // test a sender with a single trait
  static_assert(!is_typed_sender<sender_with_value_types>::value, "Expected false.");

  // test a sender with two traits
  static_assert(!is_typed_sender<sender_with_value_types_and_error_types>::value, "Expected false.");

  // test a sender with all traits
  static_assert(is_typed_sender<sender_with_all_types>::value, "Expected true.");

  // test a reference to a sender with all traits
  static_assert(is_typed_sender<sender_with_all_types&>::value, "Expected true.");

  // test a sender that specializes sender_traits
  static_assert(is_typed_sender<sender_that_specializes_sender_traits>::value, "Expected true.");

  // test a reference to a sender that specializes sender_traits
  static_assert(is_typed_sender<sender_that_specializes_sender_traits&>::value, "Expected true.");
}

