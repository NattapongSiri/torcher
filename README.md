# torcher
Port some of Tensor create/delete from Caffe2, an engine behind PyTorch into Rust.

# Objective
The main goal behind this is to make a Rust library that can prepare PyTorch data without spending the cost of data convertion from Rust->Python->C.
This project directly use Caffe2 library inside PyTorch to construct an underlying storage of Tensor.

# Installation
1. Install PyTorch 1.0 following this [guide](https://pytorch.org/get-started/locally/)
2. Locate Caffe2 lib. It can be found where you install PyTorch. For example in virtualenv in MS.Windows, it'll be found inside the directory `Lib/site-packages/torch/lib` inside directory where you create virtualenv. For Linux virtualenv example, it'll be in `lib/python3.7/site-packages/torch/lib` on virtualenv directory.
3. Add this lib as your dependency in your cargo.toml. For example: 
    ```toml
    [dependencies]
    torcher = { git="https://github.com/NattapongSiri/torcher" }
    ```
4. Create directory `clib` on your project root.
   - In MS. Windows, copy following files from Caffe2 lib into it.
      1. c10.dll
      2. caffe2.dll and caffe2.lib
      3. libiomp5md.dll
   - In Linux, copy only libcaffe2.so file
5. Set environment variable to make linker know where to find the lib
    - In MS. Windows, set `PATH` to include directory `clib`
    - In Linux, set `LD_LIBRARY_PATH` to include directory `clib`

# Use case
- populate fixed size tensor 
```Rust
extern crate torcher;

use torcher::{populate_tensor, shape};
use torcher::tensor::{ByteTensor, Tensor};
use torcher::tensor::{BasicManipulateOp, CreateOp, ViewOp};

// create 3d range tensor of shape [2, 5, 2]
// tensor will look like 
// [
//      [ // 1st of dim 0
//          [[0, 1], [2, 3]], // 1st of dim 1
//          ...
//          [[16, 17], [18, 19]]  // 5th of dim 1
//      ], 
//      [ // 2nd of dim 0
//          [[20, 21], [22, 23]], // 1st of dim 1
//          ...
//          [[36, 37], [38, 39]]  // 5th of dim 1
//      ]
// ]

// populate_tensor macro produce 1d macro, so the size is equals to product of all the dim
let tensor = populate_tensor!(u8, 2 * 5 * 2, |i, v| {
    // make range tensor from 0 to n
    *v = i as u8;
}).view(shape!(2, 5, 2)).unwrap(); // now make 3d view of tensor

// now, pass tensor back to Python to leverage all PyTorch functionalities.
```