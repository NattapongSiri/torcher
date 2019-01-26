//! This crate provides basic tensor create and edit operation.
//! If the tensor is created based on existing storage, the 
//! underlying storage will be freed when tensor is freed.
//! If user uses any of unsafe operation, it's essential to
//! keep the based tensor live as long as all other tensors
//! that got return from unsafe operation. Otherwise, it
//! result in undefined behavior.
//! 
//! All tensor types provided by this crate are:
//! - Byte tensor
//!     - [ByteTensor](struct.ByteTensor.html) - stores u8
//!     - [CharTensor](struct.CharTensor.html) - stores i8
//! - Floating point tensor
//!     - [DoubleTensor](struct.DoubleTensor.html) - stores f64
//!     - [FloatTensor](struct.FloatTensor.html) - stores f32
//! - Integer tensor
//!     - [IntTensor](struct.IntTensor.html) - stores i32
//!     - [LongTensor](struct.LongTensor.html) - stores i64
//!     - [ShortTensor](struct.ShortTensor.html) - stores i16
//! 
//! __Note:__ There's no HalfTensor for 16 bits float like Caffe2 as standard Rust
//! doesn't have such data type.
//! 
//! All tensors can be iterate by invoke `iter()` or `iter_mut()` explicitly.
//! All borrowed tensor can also be put into `for v in &tensor` or 
//! `for v in &mut tensor` because borrowed tensor implement IntoIterator trait.
//! All tensors implement Deref and DerefMut which deref into underlying storage
//! data. This allow convenient access such as tensor[0] = 1.
//! However, if you need multiple mutations and performance is concerned, calling to 
//! [data](trait.Tensor.html#tymethod.data) function then mutate the return data
//! is faster. For example: `let mut data = tensor.data_mut()` then all later mutation is
//! done on `data`. This is because deref has one additional indirection on every
//! subscription access.
//! 
//! __Important note:__ iterative style will iterate per size and stride defined
//! on tensor but deref style will give a direct access to backend data.
//! To have similar data traversal like iterator on deref data, 
//! you need to calculate index by taking size and stride into equation.
//! 
//! All Tensor derivative types implement `From<&[T]>` where `T` is it own primitive 
//! type counterpart. For example: `impl<'a> From<&'a [f32]> for FloatTensor<'a>`
//! 
//! All other structs are used for manipulating C pointer of Caffe2 tensor.
//! 
//! # Safety
//! All unsafe operation create new tensor instance but reuse existing
//! storage. If one of tensor mutate data, it'll also mutate all other
//! tensor's data.
extern crate common;
extern crate storage;
extern crate tensor_derive;

use common::THDescBuff;
use std::iter::Iterator;
use std::ops::{Deref, DerefMut};
use std::os::raw::{c_int};
use storage::{ByteStorage, CharStorage, DoubleStorage, FloatStorage, IntStorage, LongStorage, ShortStorage, TensorStorage};
use tensor_derive::TorchTensor;

/// A trait that all tensor derivative need to implemented
pub trait Tensor<T> {
    type Storage : TensorStorage;

    /// Construct an empty tensor
    fn new() -> Self;
    /// Always return a new deep clone tensor with contiguous storage according to current size.
    /// In Python, it'll return the same Tensor if it's already contiguous.
    /// So to get similar effect, consider use [is_contiguous](trait.Tensor.html#tymethod.is_contiguous)
    /// to check first if it's already contiguous.
    fn new_contiguous(&self) -> Self;
    /// Create shallow clone of Tensor that share the same storage as this tensor.
    /// This function always success unlike the same function provided by
    /// PyTorch counterpart. If underlying storage doesn't support such operation,
    /// it'll be undefined behavior.
    /// In PyTorch, if the storage doesn't support this operation, it'll raise
    /// an Exception asking user to call contiguous.
    /// See PyTorch documentation on narrow/view of tensor for more detail.
    /// 
    /// # Safety
    /// It's unsafe because the new tensor share underlying storage with this tensor.
    /// The new tensor will never free underlying storage.
    unsafe fn new_narrow(&self, dim: usize, i: usize, size: usize) -> Self;
    /// ???
    /// 
    /// # Safety
    /// It's unsafe because the new tensor share the underlying storage with this tensor.
    /// The new tensor will never free underlying storage.
    unsafe fn new_select(&self, dim: usize, i: usize) -> Self;
    /// Transpose between the two dimension and return a tranposed tensor.
    /// 
    /// # Safety
    /// The new tensor share the underlying storage with this tensor.
    /// The new tensor will never free underlying storage.
    unsafe fn new_transpose(&self, dim_1: usize, dim_2: usize) -> Self;
    /// Similar to PyTorch unfold, it'll append the new dimension to this tensor
    /// and repeatly fill in the value with value copy from given dimension.
    /// The new dimension will have size according to specified size.
    /// The original dimension will be shrunk to the original ((dimension - size) / step) + 1
    /// 
    /// # Safety
    /// New tensor share underlying storage with this tensor.
    /// The new tensor will never free underlying storage.
    unsafe fn new_unfold(&self, dim: usize, size: usize, step: usize) -> Self;
    /// Consume storage and associate it with new tensor.
    /// It map directly with Caffe2 function that responsible to do the similar task.
    /// 
    /// The underlying storage will be free awhen this tensor is drop
    fn new_with_storage_1d(store: Self::Storage, offset: usize, size: usize) -> Self;
    /// Consume storage and associate it with new tensor.
    /// It map directly with Caffe2 function that responsible to do the similar task.
    /// 
    /// The underlying storage will be free awhen this tensor is drop
    fn new_with_storage_2d(store: Self::Storage, offset: usize, size: [usize; 2], stride: usize) -> Self;
    /// Consume storage and associate it with new tensor.
    /// It map directly with Caffe2 function that responsible to do the similar task.
    /// 
    /// The underlying storage will be free awhen this tensor is drop
    fn new_with_storage_3d(store: Self::Storage, offset: usize, size: [usize; 3], stride: [usize; 2]) -> Self;
    /// Consume storage and associate it with new tensor.
    /// It map directly with Caffe2 function that responsible to do the similar task.
    /// 
    /// The underlying storage will be free awhen this tensor is drop
    fn new_with_storage_4d(store: Self::Storage, offset: usize, size: [usize; 4], stride: [usize; 3]) -> Self;
    /// Consume storage and associate it with new tensor.
    /// It map directly with Caffe2 function that responsible to do the similar task.
    /// 
    /// The underlying storage will be free awhen this tensor is drop
    fn new_with_storage_nd(store: Self::Storage, offset: usize, size: &[usize], stride: &[usize]) -> Self;
    /// Create new empty 1d tensor with contiguous stride.
    /// 
    /// The underlying storage will always automatically free by
    /// Caffe2 lib
    fn new_with_size_1d(size: usize) -> Self;
    /// Create new empty 2d tensor with contiguous stride.
    /// 
    /// The underlying storage will always automatically free by
    /// Caffe2 lib
    fn new_with_size_2d(size: [usize; 2]) -> Self;
    /// Create new empty 3d tensor with contiguous stride.
    /// 
    /// The underlying storage will always automatically free by
    /// Caffe2 lib
    fn new_with_size_3d(size: [usize; 3]) -> Self;
    /// Create new empty 4d tensor with contiguous stride.
    /// 
    /// The underlying storage will always automatically free by
    /// Caffe2 lib
    fn new_with_size_4d(size: [usize; 4]) -> Self;

    /// Return a slice of underlying data. The size of data is 
    /// exactly equals to actual data being represent by this tensor.
    /// # Examples
    /// ```Rust
    /// use torcher::storage::FloatStorage;
    /// use torcher::tensor::FloatTensor;
    /// let mut storage = FloatStorage::new_with_size(100);
    /// storage.iter_mut().enumerate().for_each(|(i, v)| *v = i as f32);
    /// let mut tensor = FloatTensor::new_with_storage_3d(storage, 2, [3, 2, 2], [2, 1]);
    /// dbg!(tensor.data().len()); // print tensor.data().len() = 7
    /// ```
    fn data(&self) -> &[T];
    /// Return a mutable slice of underlying data. The size of data is 
    /// exactly equals to actual data being represent by this tensor.
    /// # Examples
    /// ```Rust
    /// use torcher::storage::FloatStorage;
    /// use torcher::tensor::FloatTensor;
    /// let mut storage = FloatStorage::new_with_size(100);
    /// storage.iter_mut().enumerate().for_each(|(i, v)| *v = i as f32);
    /// let mut tensor = FloatTensor::new_with_storage_3d(storage, 2, [3, 2, 2], [2, 1]);
    /// dbg!(tensor.data().len()); // print tensor.data().len() = 7
    /// ```
    fn data_mut(&mut self) -> &mut [T];
    /// Just a wrapper to Caffe2 function.
    /// It currently only print size of tensor.
    fn desc(&self) -> String;
    /// Get total number of dimension this tensor is representing.
    fn dimensions(&self) -> usize;
    /// Get scalar value out of this scalar tensor.
    fn get_0d(&self) -> T;
    /// Get scalar value from given index of tensor.
    fn get_1d(&self, i: usize) -> T;
    /// Get scalar value from given indices of 2d tensor.
    fn get_2d(&self, i: [usize; 2]) -> T;
    /// Get scalar value from given indices of 3d tensor
    fn get_3d(&self, i: [usize; 3]) -> T;
    /// Get scalar value from given indices of 4d tensor
    fn get_4d(&self, i: [usize; 4]) -> T;
    /// Check if current tensor is contiguous
    fn is_contiguous(&self) -> bool;
    /// Total number of element this tensor is representing.
    /// It may not be equals to number return by `data().len()`
    fn numel(&self) -> usize;
    /// Resize into the shape similar to given tensor
    fn resize_as(&mut self, ref_tensor: &Self) {
        let ref_shape = ref_tensor.shape();

        self.resize_nd(ref_shape.0, ref_shape.1);
    }
    /// Make tensor a scalar tensor.
    /// It will be a single value, not an array.
    /// It isn't the same as 1d of size 1 as such tensor
    /// will be an array containing 1 element but 0d mean
    /// it's not array.
    fn resize_0d(&mut self);
    /// Resize tensor to new size and reset stride to 1.
    /// This will reset all view applied to this tensor.
    fn resize_1d(&mut self, size: usize);
    /// Resize tensor to new size and reset stride to [size[1], 1]
    /// This will reset all view applied to this tensor.
    fn resize_2d(&mut self, size: [usize; 2]);
    /// Resize tensor to new size and reset stride to [size[1], size[2], 1].
    /// This will reset all view applied to this tensor.
    fn resize_3d(&mut self, size: [usize; 3]);

    // resize4d and resize5d cause corrupted tensor.
    // github issue tracker url:
    // https://github.com/pytorch/pytorch/issues/16138

    // fn resize_4d(&mut self, size: [usize; 4]);
    // fn resize_5d(&mut self, size: [usize; 5]);

    /// Resize current tensor to given size and stride
    /// 
    /// # Parameter
    /// `size` need to have `size.len() == dim`
    /// `stride` need to have `stride.len() == dim - 1`
    fn resize_nd(&mut self, size: &[usize], stride: &[usize]);
    /// Set scalar tensor to specific value
    fn set_0d(&mut self, v: T);
    /// Set 1d tensor at index `i` to given `v`
    /// 
    /// This function perform index conversion base on size/stride on each call.
    fn set_1d(&mut self, i: usize, v: T);
    /// Set 2d tensor at index `i` to given `v`
    /// 
    /// This function perform index conversion base on size/stride on each call.
    fn set_2d(&mut self, i: [usize; 2], v: T);
    /// Set 3d tensor at index `i` to given `v`
    /// 
    /// This function perform index conversion base on size/stride on each call.
    fn set_3d(&mut self, i: [usize; 3], v: T);
    /// Set 4d tensor at index `i` to given `v`
    /// 
    /// This function perform index conversion base on size/stride on each call.
    fn set_4d(&mut self, i: [usize; 4], v: T);
    /// Return tuple of size and stride of this tensor
    fn shape(&self) -> (&[usize], &[usize]);
    /// Return size of given dimension of this tensor
    fn size(&self, dim: usize) -> usize;
    /// Return underlying storage.
    /// If it's empty tensor, it may be None
    fn storage(&mut self) -> &mut Option<Self::Storage>;
    /// Return storage offset of this tensor
    fn storage_offset(&self) -> usize;
    /// Return stride of given dimension of this tensor
    fn stride(&self, dim: usize) -> usize;
    /// Perform tensor squeeze. It'll flatten any dimension
    /// that have size 1
    fn squeeze(&self) -> Self;
}

/// An Iterator over tensor data.
/// It'll return a copy of underlying data, not a reference.
/// This is because returning a borrow value, when user want to read
/// data, they will need to deref it. This will have some performance
/// penalty. Since all tensor types are on primitive data. Copy it
/// prove to yield highest performance.
/// 
/// # Examples
/// ```Rust
/// use torcher::tensor::FloatTensor;
/// let mut tensor = FloatTensor::new_with_size_1d(10);
/// tensor.data_mut().iter_mut().enumerate().for_each(|(i, v)| *v = i as f32);
/// for v in &tensor {
///     dbg!(v)
/// }
/// ```
pub struct TensorIterator<'a, T>
where T: 'a
{
    cur_i: Vec<usize>,
    cur_j: usize,
    cur_offset: Vec<usize>,

    data: &'a [T],
    size: &'a [usize],
    stride: &'a [usize]
}

impl<'a, T> TensorIterator<'a, T> {
    pub fn new(data: &'a [T], size: &'a [usize], stride: &'a [usize]) -> TensorIterator<'a, T> {
        let n = size.len() - 1;
        TensorIterator {
            cur_i: vec![0; n],
            cur_j: 0,
            cur_offset: vec![0; n],

            data: data,
            size: size,
            stride: stride
        }
    }
}

impl<'a, T> Iterator for TensorIterator<'a, T>
where T: Copy
{
    type Item=T;

    fn next(&mut self) -> Option<Self::Item> {
        fn update_cursor(cur_i: &mut [usize], cur_offset: &mut [usize], n: usize, offset: usize) {
            cur_offset[n] += offset;

            // reset offset and i for those dim after the next one.
            for i in (n + 1)..cur_offset.len() {
                cur_i[i] = 0;
                cur_offset[i] = 0;
            }
        }

        let mut n = self.size.len() - 1;

        if self.cur_j < self.size[n] {
            let offset = self.cur_i.iter().enumerate().fold(0, |o, (idx, _)| {
                o + self.cur_offset[idx]
            });

            let result = Some(self.data[offset + self.cur_j]);
            self.cur_j += 1;
            result
        } else {
            n -= 1;

            while n > 0 {
                self.cur_i[n] += 1;
                if self.cur_i[n] < self.size[n] {
                    update_cursor(&mut self.cur_i, &mut self.cur_offset, n, self.stride[n]);
                    
                    break;
                } else {
                    n -= 1;
                }
            }

            if n == 0 {
                self.cur_i[n] += 1;
                if self.cur_i[n] < self.size[n] {
                    update_cursor(&mut self.cur_i, &mut self.cur_offset, n, self.stride[n]);
                } else {
                    return None;
                }
            }
            self.cur_j = 0;
            self.next()
        }
    }
}

/// A mutable Iterator over tensor data.
/// It'll return a mutable data reference.
/// Such operation is considered unsafe, thought the
/// iterator doesn't mark as unsafe. This is because signature
/// defined by Iterator doesn't have unsafe mark.
/// 
/// # Safety
/// It can be unsafe to iterating over &mut tensor.
/// It consider unsafe because it may cause datarace.
/// 
/// # Examples
/// ```Rust
/// use torcher::tensor::FloatTensor;
/// let mut tensor = FloatTensor::new_with_size_1d(10);
/// (&mut tensor).into_iter().enumerate().for_each(|(i, v)| *v = i as f32);
/// ```
pub struct TensorIterMut<'a, T>
where T: 'a
{
    cur_i: Vec<usize>,
    cur_j: usize,
    cur_offset: Vec<usize>,

    data: &'a mut [T],
    size: &'a [usize],
    stride: &'a [usize]
}

impl<'a, T> TensorIterMut<'a, T> {
    pub fn new(data: &'a mut [T], size: &'a [usize], stride: &'a [usize]) -> TensorIterMut<'a, T> {
        let n = size.len() - 1;
        TensorIterMut {
            cur_i: vec![0; n],
            cur_j: 0,
            cur_offset: vec![0; n],

            data: data,
            size: size,
            stride: stride
        }
    }
}

impl<'a, T> Iterator for TensorIterMut<'a, T> {
    type Item=&'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        fn update_cursor(cur_i: &mut [usize], cur_offset: &mut [usize], n: usize, offset: usize) {
            cur_offset[n] += offset;

            // reset offset and i for those dim after the next one.
            for i in (n + 1)..cur_offset.len() {
                cur_i[i] = 0;
                cur_offset[i] = 0;
            }
        }

        let mut n = self.size.len() - 1;
        if self.cur_j < self.size[n] {
            let offset = self.cur_i.iter().enumerate().fold(0, |o, (idx, _)| {
                o + self.cur_offset[idx]
            });
            // need unsafe raw pointer because self doesn't have lifetime but
            // return item need lifetime
            unsafe {
                let result = Some(&mut *((&mut self.data[offset + self.cur_j]) as *mut T));
                self.cur_j += 1;
                result
            }
        } else if n > 0 {
            n -= 1;

            while n > 0 {
                self.cur_i[n] += 1;
                if self.cur_i[n] < self.size[n] {
                    update_cursor(&mut self.cur_i, &mut self.cur_offset, n, self.stride[n]);
                    
                    break;
                } else {
                    n -= 1;
                }
            }

            if n == 0 {
                self.cur_i[n] += 1;
                if self.cur_i[n] < self.size[n] {
                    update_cursor(&mut self.cur_i, &mut self.cur_offset, n, self.stride[n]);
                } else {
                    return None;
                }
            }
            self.cur_j = 0;
            self.next()
        } else {
            None
        }
    }
}

#[TorchTensor(u8 = "ByteStorage")]
pub struct ByteTensor;
#[TorchTensor(i8 = "CharStorage")]
pub struct CharTensor;

#[TorchTensor(f64 = "DoubleStorage")]
pub struct DoubleTensor;
#[TorchTensor(f32 = "FloatStorage")]
pub struct FloatTensor;

#[TorchTensor(i32 = "IntStorage")]
pub struct IntTensor;
#[TorchTensor(i64 = "LongStorage")]
pub struct LongTensor;
#[TorchTensor(i16 = "ShortStorage")]
pub struct ShortTensor;

mod convert;
include!(concat!(env!("OUT_DIR"), "/conv.rs"));

#[cfg(test)]
mod tests;