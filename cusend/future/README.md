This directory collects functionality related to futures, which are objects that represent the eventual completion of a computation at some point in the *future*.

Promises are objects that represents an intention or *promise* to execute a computation.

There are two class templates:

  1. `future<T>`
  2. `host_promise<T>`

`future<T>` represents the eventual completion of a computation executing on one or many devices. It presents an interface similar to `std::future<T>`.

`host_promise<T>` represents a promise to execute a computation on a CPU and deliver the result of that computation to a device. It presents an interface similar to `std::promise<T>`.

Importantly, `host_promise::get_future()` retrieves a special type of future representing the eventual completion of a CPU computation that may be used to create dependent work on one or many devices.

