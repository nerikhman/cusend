#include <cassert>
#include <cudex/sender/is_many_receiver_of.hpp>
#include <exception>


struct my_many_receiver_of_void
{
  void set_value();

  void set_error(std::exception_ptr) &&;

  void set_done() &&;
};


struct my_many_receiver_of_int
{
  void set_value(int);

  void set_error(std::exception_ptr) &&;

  void set_done() &&;
};


struct my_many_receiver_of_either
{
  void set_value();

  void set_value(int);

  void set_error(std::exception_ptr) &&;

  void set_done() &&;
};


struct not_a_many_receiver
{
  void set_value() &&;

  void set_value(int) &&;

  void set_error(std::exception_ptr) &&;

  void set_done() &&;
};


void test_is_many_receiver_of()
{
  using namespace cudex;

  {
    // test not a many receiver
    static_assert(!is_many_receiver_of<not_a_many_receiver, void>::value, "Expected false.");
  }

  {
    // test a many receiver of void with void
    static_assert(is_many_receiver_of<my_many_receiver_of_void, void>::value, "Expected true.");
  }

  {
    // test a reference to a many receiver of void with void
    static_assert(!is_many_receiver_of<my_many_receiver_of_void&, void>::value, "Expected false.");
  }

  {
    // test an rvalue reference to a many receiver of void with void
    static_assert(is_many_receiver_of<my_many_receiver_of_void&&, void>::value, "Expected true.");
  }

  {
    // test a many receiver of void with int
    static_assert(!is_many_receiver_of<my_many_receiver_of_void, int>::value, "Expected false.");
  }

  {
    // test a many receiver of int with int
    static_assert(is_many_receiver_of<my_many_receiver_of_int, int>::value, "Expected true.");
  }

  {
    // test a reference to a many receiver of int with int
    static_assert(!is_many_receiver_of<my_many_receiver_of_int&, int>::value, "Expected false.");
  }

  {
    // test an rvalue reference to a many receiver of int with int
    static_assert(is_many_receiver_of<my_many_receiver_of_int&&, int>::value, "Expected true.");
  }

  {
    // test a many receiver of int with void
    static_assert(!is_many_receiver_of<my_many_receiver_of_int, void>::value, "Expected false.");
  }

  {
    // test a many receiver of either with void
    static_assert(is_many_receiver_of<my_many_receiver_of_either, void>::value, "Expected true.");
  }

  {
    // test a reference to a many receiver of either with void
    static_assert(!is_many_receiver_of<my_many_receiver_of_either&, void>::value, "Expected false.");
  }

  {
    // test a rvalue reference to a many receiver of either with void
    static_assert(is_many_receiver_of<my_many_receiver_of_either&&, void>::value, "Expected true.");
  }

  {
    // test a many receiver of either with int
    static_assert(is_many_receiver_of<my_many_receiver_of_either, int>::value, "Expected true.");
  }

  {
    // test a reference to a many receiver of either with int
    static_assert(!is_many_receiver_of<my_many_receiver_of_either&, int>::value, "Expected false.");
  }

  {
    // test an rvalue reference to a many receiver of either with int
    static_assert(is_many_receiver_of<my_many_receiver_of_either&&, int>::value, "Expected true.");
  }
}

