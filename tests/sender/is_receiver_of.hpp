#include <cassert>
#include <cudex/sender/is_receiver_of.hpp>
#include <exception>


struct my_receiver_of_void
{
  void set_value() &&;

  void set_error(std::exception_ptr) &&;

  void set_done() &&;
};


struct my_receiver_of_int
{
  void set_value(int) &&;

  void set_error(std::exception_ptr) &&;

  void set_done() &&;
};


struct my_receiver_of_either
{
  void set_value() &&;

  void set_value(int) &&;

  void set_error(std::exception_ptr) &&;

  void set_done() &&;
};


struct not_a_receiver {};


void test_is_receiver_of()
{
  {
    // test not a receiver
    static_assert(!cudex::is_receiver_of<not_a_receiver, void>::value, "Expected false.");
  }

  {
    // test a receiver of void with void
    static_assert(cudex::is_receiver_of<my_receiver_of_void, void>::value, "Expected true.");
  }

  {
    // test a reference to a receiver of void with void
    static_assert(cudex::is_receiver_of<my_receiver_of_void&&, void>::value, "Expected true.");
  }

  {
    // test a receiver of void with int
    static_assert(!cudex::is_receiver_of<my_receiver_of_void, int>::value, "Expected false.");
  }

  {
    // test a receiver of int with int
    static_assert(cudex::is_receiver_of<my_receiver_of_int, int>::value, "Expected true.");
  }

  {
    // test a reference to a receiver of int with int
    static_assert(cudex::is_receiver_of<my_receiver_of_int&&, int>::value, "Expected true.");
  }

  {
    // test a receiver of int with void
    static_assert(!cudex::is_receiver_of<my_receiver_of_int, void>::value, "Expected false.");
  }

  {
    // test a receiver of either with void
    static_assert(cudex::is_receiver_of<my_receiver_of_either, void>::value, "Expected true.");
  }

  {
    // test a reference to a receiver of either with void
    static_assert(cudex::is_receiver_of<my_receiver_of_either&&, void>::value, "Expected true.");
  }

  {
    // test a receiver of either with int
    static_assert(cudex::is_receiver_of<my_receiver_of_either, int>::value, "Expected true.");
  }

  {
    // test a reference to a receiver of either with int
    static_assert(cudex::is_receiver_of<my_receiver_of_either&&, int>::value, "Expected true.");
  }
}

