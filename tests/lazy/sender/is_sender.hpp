#include <cassert>
#include <cusend/sender/is_sender.hpp>


struct derived_from_sender_base : cusend::sender_base {};


struct has_sender_types
{
  template<template<class...> class Tuple, template<class...> class Variant>
  using value_types = Variant<Tuple<>>;

  template<template<class...> class Variant>
  using error_types = Variant<>;

  static constexpr bool sends_done = false;
};


struct specializes_sender_traits
{
};


namespace cusend
{


template<> struct sender_traits<specializes_sender_traits> {};

} // end cusend


struct not_a_sender {};


void test_is_sender()
{
  using namespace cusend;

  {
    // test derived_from_sender_base
    static_assert(is_sender<derived_from_sender_base>::value, "Expected a sender.");
  }


  {
    // test a reference to derived_from_sender_base
    static_assert(is_sender<derived_from_sender_base&>::value, "Expected a sender.");
  }


  {
    // test has_sender_types
    static_assert(is_sender<has_sender_types>::value, "Expected a sender.");
  }


  {
    // test a reference to has_sender_types
    static_assert(is_sender<has_sender_types&>::value, "Expected a sender.");
  }


  {
    // test specializes_sender_traits
    static_assert(is_sender<specializes_sender_traits>::value, "Expected a sender.");
  }


  {
    // test a reference to specializes_sender_traits
    static_assert(is_sender<specializes_sender_traits&>::value, "Expected a sender.");
  }

  {
    // test not a sender
    static_assert(!is_sender<not_a_sender>::value, "Expected not a sender.");
  }
}

