This directory contains the default implementations of customizable sender combinators such as `transform` and others.

For every combinator `foo`, there is a headers `default_foo.hpp`.

`default_foo.hpp` contains a function named `detail::default_foo` and a type trait `detail::default_foo_t` - the type of `default_foo(arg, args...)` result.

# Sender Guidelines

Many of the default implementations of sender combinators themselves define a sender.

The general implementation pattern of these sender types enables them to be "multi-shot" senders.

That means that when possible, operations that would otherwise consume a sender
(such as `connect`), may be called multiple times.

Technically, that means that these operations should have both `const` and `rvalue` overloads.

Moreover, good citizen senders should customize sender operations when an efficient implementation is available.

To make it concrete, consider `invoke_sender`, used by `default_invoke_on`:

    template<class Executor, class Invocable>
    class invoke_sender
    {
      public:
        // OtherInvocable is a forwarding reference, enabling both copyable and move-only invocables
        template<class OtherInvocable>
        invoke_sender(const Executor& ex, OtherInvocable&& invocable);


        // allow copies of invoke_sender when possible
        invoke_sender(const invoke_sender&) = default;

        // move-construction must always be possible for senders
        invoke_sender(invoke_sender&&) = default;


        // consuming connect() must always be possible for senders
        template<class Receiver>
        auto connect(Receiver&& r) &&;

        // enable non-consuming connect() when Invocable is copyable
        template<class Receiver,
                 CUSEND_REQUIRES(std::is_copy_constructible<Invocable>::value)
                >
        auto connect(Receiver&& r) const &;


        // to be a good citizen, customize consuming on()
        template<class OtherExecutor>
        invoke_sender<OtherExecutor, Invocable> on(const OtherExecutor& ex) &&;

        // enable non-consuming on() when Invocable is copyable
        template<class OtherExecutor,
                 CUSEND_REQUIRES(std::is_copy_constructible<Invocable>::value)
                >
        invoke_sender<OtherExecutor, Invocable> on(const OtherExecutor& ex) const &;
    };

To summarize:

  * Senders should receive forwarding references to internal state objects to enable both move-only and copyable members.
  * Senders must provide a move constructor.
  * Senders should provide a copy constructor.
  * Senders must provide rvalue reference `connect`.
  * Senders should provide a `const` reference `connect` when internal state is copyable.
  * Senders should provide rvalue refeference and `const` reference customizations of other sender operations when an efficient implementation is available.

