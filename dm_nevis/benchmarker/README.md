# Benchmarker

The benchmarker library is a platform-agnostic collection of modules for
training and evaluating the performance of learners as they interact with long
task streams.

This package defines the key APIs that learners and streams must satisfy to be
evaluable using the Nevis evaluation protocol.

## Streams

The most important stream can be found at `streams/nevis_stream.py`. This stream
contains the 100+ dataset image classification stream reported in the paper.

## Learners

The key learner interface can be found at
`learners/learner_interface.py`. This is the interface that learners
must satisfy in order to for the benchmarker to evaluate them.
