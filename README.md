# LSTM in C++

This is an implementation of an LSTM in C++. It works enough to reduce the error over time, but it doesn't do much more. It was just an exercise for me to learn more about what happens inside of TensorFlow or any of the other frameworks.

It's a simple character-level language model, trained on the combined works of Arthur Conan Doyle.

There are many things wrong with this as it stands.
 * I'm predicting the next character at every step, but I didn't implement softmax.
 * It copies huge arrays around all the time. I spent no time optimizing this. As a result, on my MacBook, it takes ober 8 hours to get through the whole input text.
 * Learning rate is fixed at 0.1.
 * It gets stuck in local minima. There are no restarts.
 * Initialization is all wrong.
