# torcher
Port some of Tensor create/delete from Caffe2, an engine behind PyTorch into Rust.

# Objective
The main goal behind this is to make a Rust library that can prepare PyTorch data without spending the cost of data convertion from Rust->Python->C.
This project directly use Caffe2 library inside PyTorch to construct an underlying storage of Tensor.
