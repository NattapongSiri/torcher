#![feature(prelude_import)]
#![no_std]
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
//! storage by raw pointer. If one of tensor mutate data, it'll also
//! mutate all other tensor's data. It requires that only one
//! tensor responsible for dropping storage. Failure to do so will
//! result in undefined behavior.
#[prelude_import]
use ::std::prelude::v1::*;
#[macro_use]
extern crate std as std;
extern crate common;
extern crate storage;
extern crate tensor_derive;

use core::ops::Range;
use common::THDescBuff;
use std::cmp;
use std::iter::Iterator;
use std::ops::{Deref, DerefMut};
use std::os::raw::{c_int};
use storage::{ByteStorage, CharStorage, DoubleStorage, FloatStorage,
              IntStorage, LongStorage, ShortStorage, TensorStorage};
use tensor_derive::TorchTensor;


#[cfg(feature = "safe")]
use std::cell::RefCell;
#[cfg(feature = "safe")]
use std::rc::Rc;

/// Basic tensor operation for simple data manipulation.
/// This includes data read/write operation and tensor shape
/// related operation.
pub trait BasicManipulateOp<S: TensorStorage> {
    type
    Datum;

    /// Just a wrapper to Caffe2 function.
    /// It currently only print size of tensor.
    fn desc(&self)
    -> String;
    /// Get total number of dimension this tensor is representing.
    fn dimensions(&self)
    -> usize;
    /// Get scalar value out of this scalar tensor.
    fn get_0d(&self)
    -> Self::Datum;
    /// Get scalar value from given index of tensor.
    fn get_1d(&self, i: usize)
    -> Self::Datum;
    /// Get scalar value from given indices of 2d tensor.
    fn get_2d(&self, i: [usize; 2])
    -> Self::Datum;
    /// Get scalar value from given indices of 3d tensor
    fn get_3d(&self, i: [usize; 3])
    -> Self::Datum;
    /// Get scalar value from given indices of 4d tensor
    fn get_4d(&self, i: [usize; 4])
    -> Self::Datum;
    /// Check if current tensor is contiguous
    fn is_contiguous(&self)
    -> bool;
    /// Convenient method to get an iterator of tensor according
    /// to size and stride
    fn iter(&self)
    -> TensorIterator<Self::Datum>;
    /// Convenient method to get a mutable iterator of tensor according
    /// to size and stride
    fn iter_mut(&mut self)
    -> TensorIterMut<Self::Datum>;
    /// Total number of element this tensor is representing.
    /// It may not be equals to number return by `data().len()`
    fn numel(&self)
    -> usize;
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
    fn set_0d(&mut self, v: Self::Datum);
    /// Set 1d tensor at index `i` to given `v`
    /// 
    /// This function perform index conversion base on size/stride on each call.
    fn set_1d(&mut self, i: usize, v: Self::Datum);
    /// Set 2d tensor at index `i` to given `v`
    /// 
    /// This function perform index conversion base on size/stride on each call.
    fn set_2d(&mut self, i: [usize; 2], v: Self::Datum);
    /// Set 3d tensor at index `i` to given `v`
    /// 
    /// This function perform index conversion base on size/stride on each call.
    fn set_3d(&mut self, i: [usize; 3], v: Self::Datum);
    /// Set 4d tensor at index `i` to given `v`
    /// 
    /// This function perform index conversion base on size/stride on each call.
    fn set_4d(&mut self, i: [usize; 4], v: Self::Datum);
    /// Return tuple of size and stride of this tensor
    fn shape(&self)
    -> (&[usize], &[usize]);
    /// Return size of given dimension of this tensor
    fn size(&self, dim: usize)
    -> usize;
    /// Return underlying storage.
    /// If it's empty tensor, it may be None
    #[cfg(feature = "safe")]
    fn storage(&mut self)
    -> &mut Option<Rc<RefCell<S>>>;
    /// Return storage offset of this tensor
    fn storage_offset(&self)
    -> usize;
    /// Return stride of given dimension of this tensor
    fn stride(&self, dim: usize)
    -> usize;
}



/// Tensor create operation.
#[cfg(feature = "safe")]
pub trait CreateOp<S: TensorStorage> {
    type
    Datum;

    /// Construct an empty tensor
    fn new()
    -> Self;
    /// Always return a new deep clone tensor with contiguous storage according to current size.
    /// In Python, it'll return the same Tensor if it's already contiguous.
    /// So to get similar effect, consider use [is_contiguous](trait.Tensor.html#tymethod.is_contiguous)
    /// to check first if it's already contiguous.
    fn new_contiguous(&self)
    -> Self;
    /// Create shallow clone of Tensor that share the same storage as this tensor.
    /// This function always success unlike the same function provided by
    /// PyTorch counterpart. If underlying storage doesn't support such operation,
    /// it'll be undefined behavior.
    /// In PyTorch, if the storage doesn't support this operation, it'll raise
    /// an Exception asking user to call contiguous.
    /// See PyTorch documentation on narrow/view of tensor for more detail.
    fn new_narrow(&self, dim: usize, i: usize, size: usize)
    -> Self;
    /// ???
    fn new_select(&self, dim: usize, i: usize)
    -> Self;
    /// Transpose between the two dimension and return a tranposed tensor.
    fn new_transpose(&self, dim_1: usize, dim_2: usize)
    -> Self;
    fn new_unfold(&self, dim: usize, size: usize, step: usize)
    -> Self;
    fn new_with_storage_1d(store: Rc<RefCell<S>>, offset: usize, size: usize)
    -> Self;
    /// Create new tensor from shared storage.
    /// 
    /// The underlying storage will be free when last tensor that use
    /// shared storage is drop
    fn new_with_storage_2d(store: Rc<RefCell<S>>, offset: usize,
                           size: [usize; 2], stride: usize)
    -> Self;
    /// Create new tensor from shared storage.
    /// 
    /// The underlying storage will be free when last tensor that use
    /// shared storage is drop
    fn new_with_storage_3d(store: Rc<RefCell<S>>, offset: usize,
                           size: [usize; 3], stride: [usize; 2])
    -> Self;
    /// Create new tensor from shared storage.
    /// 
    /// The underlying storage will be free when last tensor that use
    /// shared storage is drop
    fn new_with_storage_4d(store: Rc<RefCell<S>>, offset: usize,
                           size: [usize; 4], stride: [usize; 3])
    -> Self;
    /// Create new tensor from shared storage.
    /// 
    /// The underlying storage will be free when last tensor that use
    /// shared storage is drop
    fn new_with_storage_nd(store: Rc<RefCell<S>>, offset: usize,
                           size: &[usize], stride: &[usize])
    -> Self;
    /// Create new empty 1d tensor with contiguous stride.
    /// 
    /// The underlying storage will always automatically free by
    /// Caffe2 lib
    fn new_with_size_1d(size: usize)
    -> Self;
    /// Create new empty 2d tensor with contiguous stride.
    /// 
    /// The underlying storage will always automatically free by
    /// Caffe2 lib
    fn new_with_size_2d(size: [usize; 2])
    -> Self;
    /// Create new empty 3d tensor with contiguous stride.
    /// 
    /// The underlying storage will always automatically free by
    /// Caffe2 lib
    fn new_with_size_3d(size: [usize; 3])
    -> Self;
    /// Create new empty 4d tensor with contiguous stride.
    /// 
    /// The underlying storage will always automatically free by
    /// Caffe2 lib
    fn new_with_size_4d(size: [usize; 4])
    -> Self;
}



/// A trait that all concrete tensor derivative need to implemented
pub trait Tensor: BasicManipulateOp<<Self as Tensor>::Storage, Datum =
 <Self as Tensor>::Datum> + CreateOp<<Self as Tensor>::Storage, Datum =
 <Self as Tensor>::Datum> + ViewOp<Self> where Self: Sized {
    type
    Datum;
    type
    Storage: TensorStorage<Datum
    =
    <Self as Tensor>::Datum>;
}

trait UtilityOp<S>: BasicManipulateOp<S> where S: TensorStorage {
    /// This function will convert a None entry in size into numeric.
    /// It'll return Err if the there's more than one None inside `sizes`.
    /// It'll return Err if the new sizes and old sizes have different number
    /// of elements and there's no single `None` element.
    /// It'll also panic if all the specified size use up all elements of 
    /// the tensor and there's one `None` element.
    /// It'll panic if one of the size is 0.
    /// A direct port from Caffe2 [infer_size function](https://github.com/pytorch/pytorch/blob/21907b6ba2a37afff8f23111b4f98a83fe2b093d/aten/src/ATen/InferSize.h#L12) into Rust
    fn infer_size(&self, sizes: &[Option<usize>])
     -> Result<Vec<usize>, SizeInferError> {
        let numel = self.numel();
        let mut res: Vec<usize> = Vec::with_capacity(sizes.len());
        let mut new_size = 1;
        let mut infer_dim = None;
        for (i, s) in sizes.iter().enumerate() {
            if let Some(v) = s {
                res.push(*v);
                new_size *= v;
            } else {
                if infer_dim.is_some() {
                    return Err(SizeInferError::MultipleUnsizedError);
                }
                infer_dim = Some(i);
                res.push(1);
            }
        }

        if numel == new_size || (new_size > 0 && numel % new_size == 0) {
            if let Some(dim) = infer_dim { res[dim] = numel / new_size; }

            return Ok(res);
        }

        return Err(SizeInferError::ElementSizeMismatch);
    }

    /// Compute stride based on current size and stride for new given size.
    /// 
    /// This method is a direct port from Caffe2 API
    /// [compute_stride function](https://github.com/pytorch/pytorch/blob/21907b6ba2a37afff8f23111b4f98a83fe2b093d/aten/src/TH/THTensor.cpp#L93)
    fn compute_stride(&self, sizes: &[usize])
     -> Result<Vec<usize>, StrideComputeError> {
        let (old_size, old_stride) = self.shape();
        let mut new_stride = Vec::with_capacity(sizes.len());

        if old_size.is_empty() {
            new_stride.iter_mut().for_each(|s| *s = 1);
            return Ok(new_stride);
        }

        let numel = self.numel();
        if numel == 0 && old_size == sizes {
            return Ok(old_stride.to_owned());
        }

        if numel == 0 {
            for view_d in (0..sizes.len()).rev() {
                if view_d == sizes.len() - 1 {
                    new_stride.push(1);
                } else {
                    new_stride.push(cmp::max(sizes[view_d + 1], 1) *
                                        new_stride[sizes.len() - view_d - 1]);
                }
            }

            new_stride.reverse();
            return Ok(new_stride);
        }

        let mut view_d = sizes.len() - 1;
        // stride for each subspace in the chunk
        let mut chunk_base_stride = *old_stride.last().unwrap();
        // numel in current chunk
        let mut tensor_numel = 1;
        let mut view_numel = 1;

        for tensor_d in (0..old_size.len()).rev() {
            tensor_numel *= old_size[tensor_d];
            // if end of tensor size chunk, check view
            if tensor_d == 0 ||
                   (old_size[tensor_d - 1] != 1 &&
                        old_stride[tensor_d - 1] !=
                            tensor_numel * chunk_base_stride) {
                while view_d > 0 &&
                          (view_numel < tensor_numel || sizes[view_d] == 1) {
                    new_stride.push(view_numel * chunk_base_stride);
                    view_numel *= sizes[view_d];
                    view_d -= 1;
                }

                if view_d == 0 {
                    new_stride.push(view_numel * chunk_base_stride);
                    view_numel *= sizes[view_d];
                }
                if view_numel != tensor_numel {
                    return Err(StrideComputeError{});
                }
                if tensor_d > 0 {
                    chunk_base_stride = old_stride[tensor_d - 1];
                    tensor_numel = 1;
                    view_numel = 1;
                }
            }
        }

        if view_d != 0 { return Err(StrideComputeError{}); }

        new_stride.reverse();

        return Ok(new_stride);
    }
}

/// View related operation
pub trait ViewOp<T: Tensor> {
    /// Create a narrower view of tensor based on given range.
    /// The range end must be exclusive, e.g. `0..3`.
    /// The inclusive range end is not support, e.g. `0..=2`.
    /// Partial range isn't supported, e.g. `..3`, `2..` .
    /// Full range isn't supported, e.g. `..`.
    /// 
    /// # Example
    /// ```Rust
    /// tensor.narrow(&[0..2, 3..5 , 1..3])
    /// ```
    /// Create a narrower view of tensor based on given range.
    /// The range end must be exclusive, e.g. `0..3`.
    /// The inclusive range end is not support, e.g. `0..=2`.
    /// Partial range isn't supported, e.g. `..3`, `2..` .
    /// Full range isn't supported, e.g. `..`.
    /// 
    /// # Example
    /// ```Rust
    /// tensor.narrow(&[0..2, 3..5 , 1..3])
    /// ```
    #[cfg(feature = "safe")]
    fn narrow(&self, bound: &[Range<usize>])
    -> Result<T, NarrowError>;
    /// Apply narrow on specific dimension and return a narrowed view on
    /// given tensor along with the original tensor.
    #[cfg(feature = "safe")]
    fn narrow_on(&self, dim: usize, new_bound: Range<usize>)
    -> Result<T, NarrowError>;

    /// Perform tensor squeeze. It'll flatten any dimension
    /// that have size 1 and return the new squeezed TensorView
    #[cfg(feature = "safe")]
    fn squeeze(&self)
    -> T;
    /// Narrow down a view as per new given bound.
    /// The size need to be smaller than current size.
    /// There's strict rule on when is it possible to create view.
    /// Simplest case is if tensor is contiguous, it is possible to create view.
    /// See PyTorch document on [view](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view)
    /// for more detail.
    /// 
    /// # Example
    /// ```Rust
    /// let tensor = FloatTensor::new_with_size_1d(20);
    /// tensor.iter_mut().enumerate().for_each(|(i, v)| *v = i as f32);
    /// let view_1 = tensor.view(&[Some(2), None])?; // shape will be [2, 10]
    /// // since TensorView implement deref into Tensor, we can subview it.
    /// let view_2 = view_1.view(&[Some(5), None])?; // shape is now [5, 4]
    /// // There is utility macro to help construct view shape like this
    /// let view_3 = view_2.view(shape!([5, 2, -1]))?; // shape is now [5, 2, 2]
    /// ```
    /// 
    /// # Parameters
    /// - `bound: &[Option<(usize, usize)>]` - A slice contain tuple of (min, max) value
    #[cfg(feature = "safe")]
    fn view(&self, bound: &[Option<usize>])
    -> Result<T, ViewError>;

}

pub enum SizeInferError {

    /// Multiple None elements given to infer_size.
    /// It need to have at most 1 None element.
    MultipleUnsizedError,

    /// The number of elements of tensor isn't compatible with
    /// given size. The new size may either too small or too large.
    /// The new size must use all elements of the tensor.
    ElementSizeMismatch,
}
#[automatically_derived]
#[allow(unused_qualifications)]
impl ::std::fmt::Debug for SizeInferError {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match (&*self,) {
            (&SizeInferError::MultipleUnsizedError,) => {
                let mut debug_trait_builder =
                    f.debug_tuple("MultipleUnsizedError");
                debug_trait_builder.finish()
            }
            (&SizeInferError::ElementSizeMismatch,) => {
                let mut debug_trait_builder =
                    f.debug_tuple("ElementSizeMismatch");
                debug_trait_builder.finish()
            }
        }
    }
}

/// Cannot compute stride based on given size.
/// Either one of new view span over noncontiguous dimension
/// or view size exceed number of element.
pub struct StrideComputeError {
}
#[automatically_derived]
#[allow(unused_qualifications)]
impl ::std::fmt::Debug for StrideComputeError {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match *self {
            StrideComputeError {  } => {
                let mut debug_trait_builder =
                    f.debug_struct("StrideComputeError");
                debug_trait_builder.finish()
            }
        }
    }
}

/// Narrow operation fail.
/// Potential cause is one of range end is out of bound
pub struct NarrowError {
    dim: usize,
}
#[automatically_derived]
#[allow(unused_qualifications)]
impl ::std::fmt::Debug for NarrowError {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match *self {
            NarrowError { dim: ref __self_0_0 } => {
                let mut debug_trait_builder = f.debug_struct("NarrowError");
                let _ = debug_trait_builder.field("dim", &&(*__self_0_0));
                debug_trait_builder.finish()
            }
        }
    }
}

/// Cannot make a view from given sizes
pub enum ViewError {

    /// One of size infer related error occur
    SizeErr(SizeInferError),

    /// Stride computation error
    StrideErr(StrideComputeError),
}
#[automatically_derived]
#[allow(unused_qualifications)]
impl ::std::fmt::Debug for ViewError {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match (&*self,) {
            (&ViewError::SizeErr(ref __self_0),) => {
                let mut debug_trait_builder = f.debug_tuple("SizeErr");
                let _ = debug_trait_builder.field(&&(*__self_0));
                debug_trait_builder.finish()
            }
            (&ViewError::StrideErr(ref __self_0),) => {
                let mut debug_trait_builder = f.debug_tuple("StrideErr");
                let _ = debug_trait_builder.field(&&(*__self_0));
                debug_trait_builder.finish()
            }
        }
    }
}
impl From<SizeInferError> for ViewError {
    fn from(err: SizeInferError) -> Self { ViewError::SizeErr(err) }
}

impl From<StrideComputeError> for ViewError {
    fn from(err: StrideComputeError) -> Self { ViewError::StrideErr(err) }
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
pub struct TensorIterator<'a, T> where T: 'a {
    cur_i: Vec<usize>,
    cur_j: usize,
    cur_offset: Vec<usize>,

    data: &'a [T],
    size: &'a [usize],
    stride: &'a [usize],
}

impl <'a, T> TensorIterator<'a, T> {
    pub fn new(data: &'a [T], size: &'a [usize], stride: &'a [usize])
     -> TensorIterator<'a, T> {
        let n = size.len() - 1;
        TensorIterator{cur_i:



                           // reset offset and i for those dim after the next one.













                           // reset offset and i for those dim after the next one.

                           // need unsafe raw pointer because self doesn't have lifetime but
                           // return item need lifetime





















                           ::alloc::vec::from_elem(0, n),
                       cur_j: 0,
                       cur_offset: ::alloc::vec::from_elem(0, n),
                       data: data,
                       size: size,
                       stride: stride,}
    }
}
impl <'a, T> Iterator for TensorIterator<'a, T> where T: Copy {
    type
    Item
    =
    T;
    fn next(&mut self) -> Option<Self::Item> {
        fn update_cursor(cur_i: &mut [usize], cur_offset: &mut [usize],
                         n: usize, offset: usize) {
            cur_offset[n] += offset;
            for i in (n + 1)..cur_offset.len() {
                cur_i[i] = 0;
                cur_offset[i] = 0;
            }
        }
        let mut n = self.size.len() - 1;
        if self.cur_j < self.size[n] {
            let offset =
                self.cur_i.iter().enumerate().fold(0,
                                                   |o, (idx, _)|
                                                       {
                                                           o +
                                                               self.cur_offset[idx]
                                                       });
            let result = Some(self.data[offset + self.cur_j]);
            self.cur_j += 1;
            result
        } else {
            n -= 1;
            while n > 0 {
                self.cur_i[n] += 1;
                if self.cur_i[n] < self.size[n] {
                    update_cursor(&mut self.cur_i, &mut self.cur_offset, n,
                                  self.stride[n]);
                    break ;
                } else { n -= 1; }
            }
            if n == 0 {
                self.cur_i[n] += 1;
                if self.cur_i[n] < self.size[n] {
                    update_cursor(&mut self.cur_i, &mut self.cur_offset, n,
                                  self.stride[n]);
                } else { return None; }
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
pub struct TensorIterMut<'a, T> where T: 'a {
    cur_i: Vec<usize>,
    cur_j: usize,
    cur_offset: Vec<usize>,
    data: &'a mut [T],
    size: &'a [usize],
    stride: &'a [usize],
}
impl <'a, T> TensorIterMut<'a, T> {
    pub fn new(data: &'a mut [T], size: &'a [usize], stride: &'a [usize])
     -> TensorIterMut<'a, T> {
        let n = size.len() - 1;
        TensorIterMut{cur_i: ::alloc::vec::from_elem(0, n),
                      cur_j: 0,
                      cur_offset: ::alloc::vec::from_elem(0, n),
                      data: data,
                      size: size,
                      stride: stride,}
    }
}
impl <'a, T> Iterator for TensorIterMut<'a, T> {
    type
    Item
    =
    &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        fn update_cursor(cur_i: &mut [usize], cur_offset: &mut [usize],
                         n: usize, offset: usize) {
            cur_offset[n] += offset;
            for i in (n + 1)..cur_offset.len() {
                cur_i[i] = 0;
                cur_offset[i] = 0;
            }
        }
        let mut n = self.size.len() - 1;
        if self.cur_j < self.size[n] {
            let offset =
                self.cur_i.iter().enumerate().fold(0,
                                                   |o, (idx, _)|
                                                       {
                                                           o +
                                                               self.cur_offset[idx]
                                                       });
            unsafe {
                let result =
                    Some(&mut *((&mut self.data[offset + self.cur_j]) as
                                    *mut T));
                self.cur_j += 1;
                result
            }
        } else if n > 0 {
            n -= 1;
            while n > 0 {
                self.cur_i[n] += 1;
                if self.cur_i[n] < self.size[n] {
                    update_cursor(&mut self.cur_i, &mut self.cur_offset, n,
                                  self.stride[n]);
                    break ;
                } else { n -= 1; }
            }
            if n == 0 {
                self.cur_i[n] += 1;
                if self.cur_i[n] < self.size[n] {
                    update_cursor(&mut self.cur_i, &mut self.cur_offset, n,
                                  self.stride[n]);
                } else { return None; }
            }
            self.cur_j = 0;
            self.next()
        } else { None }
    }
}
use storage::THByteStorage;
#[repr(C)]
pub struct THByteTensor;
#[link(name = "caffe2")]
extern "C" {
    fn THByteTensor_new() -> *mut THByteTensor;
    fn THByteTensor_newClone(org: *const THByteTensor) -> *mut THByteTensor;
    fn THByteTensor_newContiguous(org: *const THByteTensor)
     -> *mut THByteTensor;
    fn THByteTensor_newNarrow(org: *const THByteTensor, dim: c_int, i: i64,
                              size: i64) -> *mut THByteTensor;
    fn THByteTensor_newSelect(org: *const THByteTensor, dim: c_int, idx: i64)
     -> *mut THByteTensor;
    fn THByteTensor_newTranspose(org: *const THByteTensor, dim_1: c_int,
                                 dim_2: c_int) -> *mut THByteTensor;
    fn THByteTensor_newUnfold(org: *const THByteTensor, dim: c_int, size: i64,
                              step: i64) -> *mut THByteTensor;
    fn THByteTensor_newWithTensor(org: *const THByteTensor)
     -> *mut THByteTensor;
    fn THByteTensor_newWithStorage1d(store: *mut THByteStorage, offset: usize,
                                     size: usize, stride: usize)
     -> *mut THByteTensor;
    fn THByteTensor_newWithStorage2d(store: *mut THByteStorage, offset: usize,
                                     size_1: usize, stride_1: usize,
                                     size_2: usize, stride_2: usize)
     -> *mut THByteTensor;
    fn THByteTensor_newWithStorage3d(store: *mut THByteStorage, offset: usize,
                                     size_1: usize, stride_1: usize,
                                     size_2: usize, stride_2: usize,
                                     size_3: usize, stride_3: usize)
     -> *mut THByteTensor;
    fn THByteTensor_newWithStorage4d(store: *mut THByteStorage, offset: usize,
                                     size_1: usize, stride_1: usize,
                                     size_2: usize, stride_2: usize,
                                     size_3: usize, stride_3: usize,
                                     size_4: usize, stride_4: usize)
     -> *mut THByteTensor;
    fn THByteTensor_newWithSize1d(size: i64) -> *mut THByteTensor;
    fn THByteTensor_newWithSize2d(size_1: i64, size_2: i64)
     -> *mut THByteTensor;
    fn THByteTensor_newWithSize3d(size_1: i64, size_2: i64, size_3: i64)
     -> *mut THByteTensor;
    fn THByteTensor_newWithSize4d(size_1: i64, size_2: i64, size_3: i64,
                                  size_4: i64) -> *mut THByteTensor;
    fn THByteTensor_free(tensor: *mut THByteTensor);
    fn THByteTensor_data(tensor: *mut THByteTensor) -> *mut u8;
    fn THByteTensor_desc(tensor: *mut THByteTensor) -> THDescBuff;
    fn THByteTensor_nDimension(tensor: *const THByteTensor) -> c_int;
    fn THByteTensor_isContiguous(tensor: *const THByteTensor) -> c_int;
    fn THByteTensor_get0d(tensor: *const THByteTensor) -> u8;
    fn THByteTensor_get1d(tensor: *const THByteTensor, i: i64) -> u8;
    fn THByteTensor_get2d(tensor: *const THByteTensor, i: i64, j: i64) -> u8;
    fn THByteTensor_get3d(tensor: *const THByteTensor, i: i64, j: i64, k: i64)
     -> u8;
    fn THByteTensor_get4d(tensor: *const THByteTensor, i: i64, j: i64, k: i64,
                          l: i64) -> u8;
    fn THByteTensor_numel(tensor: *const THByteTensor) -> usize;
    fn THByteTensor_resize0d(tensor: *mut THByteTensor);
    fn THByteTensor_resize1d(tensor: *mut THByteTensor, size_1: i64);
    fn THByteTensor_resize2d(tensor: *mut THByteTensor, size_1: i64,
                             size_2: i64);
    fn THByteTensor_resize3d(tensor: *mut THByteTensor, size_1: i64,
                             size_2: i64, size_3: i64);
    fn THByteTensor_resize4d(tensor: *mut THByteTensor, size_1: i64,
                             size_2: i64, size_3: i64, size_4: i64);
    fn THByteTensor_resize5d(tensor: *mut THByteTensor, size_1: i64,
                             size_2: i64, size_3: i64, size_4: i64,
                             size_5: i64);
    fn THByteTensor_resizeNd(tensor: *mut THByteTensor, dim: c_int,
                             size: *const i64, stride: *const i64);
    fn THByteTensor_set0d(tensor: *const THByteTensor, v: u8);
    fn THByteTensor_set1d(tensor: *const THByteTensor, i: i64, v: u8);
    fn THByteTensor_set2d(tensor: *const THByteTensor, i: i64, j: i64, v: u8);
    fn THByteTensor_set3d(tensor: *const THByteTensor, i: i64, j: i64, k: i64,
                          v: u8);
    fn THByteTensor_set4d(tensor: *const THByteTensor, i: i64, j: i64, k: i64,
                          l: i64, v: u8);
    fn THByteTensor_size(tensor: *const THByteTensor, dim: c_int) -> i64;
    fn THByteTensor_setStorageNd(tensor: *const THByteTensor,
                                 storage: THByteStorage, offset: i64,
                                 dim: c_int, size: *const i64,
                                 stride: *const i64);
    fn THByteTensor_storage(tensor: *mut THByteTensor) -> *mut THByteStorage;
    fn THByteTensor_storageOffset(tensor: *const THByteTensor) -> i64;
    fn THByteTensor_stride(tensor: *const THByteTensor, dim: c_int) -> i64;
    fn THByteTensor_squeeze(tensor: *mut THByteTensor,
                            src: *const THByteTensor);
}
pub struct ByteTensor {
    forget: bool,
    #[cfg(feature = "safe")]
    storage: Option<Rc<RefCell<ByteStorage>>>,
    tensor: *mut THByteTensor,
    size: Vec<usize>,
    stride: Vec<usize>,
}
impl ByteTensor {
    #[doc = r" Get short description of storage."]
    #[doc = r" This includes name of storage, size, and"]
    #[doc = r" sample data if it has more than 20 elements."]
    #[doc = r" If it has less than 20 elements, it'll display"]
    #[doc = r" every elements."]
    fn short_desc(&mut self) -> String {
        #[cfg(feature = "safe")]
        fn get_data(s: &ByteTensor) -> &ByteStorage {
            s.storage().as_ref().unwrap().borrow()
        }
        let size = self.size.as_slice();
        let stride = self.stride.as_slice();
        let data = get_data(self);
        let name = "ByteTensor";
        if size.iter().fold(0, |cum, v| cum + v) > 20 {
            ::alloc::fmt::format(::std::fmt::Arguments::new_v1(&["", ":size=",
                                                                 ":stride=",
                                                                 ":first(10)=",
                                                                 ":last(10)="],
                                                               &match (&name,
                                                                       &size,
                                                                       &stride,
                                                                       &&(&*data)[0..10],
                                                                       &&(&*data)[(data.len()
                                                                                       -
                                                                                       10)..data.len()])
                                                                    {
                                                                    (arg0,
                                                                     arg1,
                                                                     arg2,
                                                                     arg3,
                                                                     arg4) =>
                                                                    [::std::fmt::ArgumentV1::new(arg0,
                                                                                                 ::std::fmt::Display::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg1,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg2,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg3,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg4,
                                                                                                 ::std::fmt::Debug::fmt)],
                                                                }))
        } else {
            ::alloc::fmt::format(::std::fmt::Arguments::new_v1(&["", ":size=",
                                                                 ":stride=",
                                                                 ":data="],
                                                               &match (&name,
                                                                       &size,
                                                                       &stride,
                                                                       &data.iter())
                                                                    {
                                                                    (arg0,
                                                                     arg1,
                                                                     arg2,
                                                                     arg3) =>
                                                                    [::std::fmt::ArgumentV1::new(arg0,
                                                                                                 ::std::fmt::Display::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg1,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg2,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg3,
                                                                                                 ::std::fmt::Debug::fmt)],
                                                                }))
        }
    }
    #[doc = r" Alias for short_desc"]
    #[inline(always)]
    fn to_string(&mut self) -> String { self.short_desc() }
}
impl CreateOp<ByteStorage> for ByteTensor {
    type
    Datum
    =
    u8;
    #[cfg(feature = "safe")]
    fn new() -> ByteTensor {
        unsafe {
            ByteTensor{forget: false,
                       storage: None,
                       tensor: THByteTensor_new(),
                       size: Vec::new(),
                       stride: Vec::new(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_contiguous(&self) -> ByteTensor {
        unsafe {
            let cont = THByteTensor_newContiguous(self.tensor);
            let storage =
                Some(Rc::new(RefCell::new(ByteStorage::from(THByteTensor_storage(cont)))));
            let stride: Vec<usize> =
                (0..THByteTensor_nDimension(cont)).map(|i|
                                                           {
                                                               THByteTensor_stride(cont,
                                                                                   i
                                                                                       as
                                                                                       i32)
                                                                   as usize
                                                           }).collect();
            ByteTensor{forget: false,
                       storage: storage,
                       tensor: cont,
                       size: self.size.to_owned(),
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_narrow(&self, dim: usize, i: usize, size: usize) -> ByteTensor {
        unsafe {
            let tensor =
                THByteTensor_newNarrow(self.tensor, dim as c_int, i as i64,
                                       size as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THByteTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THByteTensor_stride(tensor, i as i32)
                                             as usize;
                                     let cur_size =
                                         THByteTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            ByteTensor{forget: false,
                       storage: storage,
                       tensor: tensor,
                       size: size,
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_select(&self, dim: usize, i: usize) -> ByteTensor {
        unsafe {
            let tensor =
                THByteTensor_newSelect(self.tensor, dim as c_int, i as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THByteTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THByteTensor_stride(tensor, i as i32)
                                             as usize;
                                     let cur_size =
                                         THByteTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            ByteTensor{forget: false,
                       storage: storage,
                       tensor: tensor,
                       size: size,
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_transpose(&self, dim_1: usize, dim_2: usize) -> ByteTensor {
        unsafe {
            let tensor =
                THByteTensor_newTranspose(self.tensor, dim_1 as c_int,
                                          dim_2 as c_int);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THByteTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THByteTensor_stride(tensor, i as i32)
                                             as usize;
                                     let cur_size =
                                         THByteTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            ByteTensor{forget: false,
                       storage: storage,
                       tensor: tensor,
                       size: size,
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_unfold(&self, dim: usize, size: usize, step: usize) -> ByteTensor {
        unsafe {
            let tensor =
                THByteTensor_newUnfold(self.tensor, dim as c_int, size as i64,
                                       step as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THByteTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THByteTensor_stride(tensor, i as i32)
                                             as usize;
                                     let cur_size =
                                         THByteTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            ByteTensor{forget: false,
                       storage: storage,
                       tensor: tensor,
                       size: size,
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_1d(store: Rc<RefCell<ByteStorage>>, offset: usize,
                           size: usize) -> ByteTensor {
        unsafe {
            let tensor =
                THByteTensor_newWithStorage1d(store.borrow_mut().storage(),
                                              offset, size, 1);
            ByteTensor{forget: false,
                       storage: Some(store),
                       tensor: tensor,
                       size: <[_]>::into_vec(box [size]),
                       stride: <[_]>::into_vec(box [1]),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_2d(store: Rc<RefCell<ByteStorage>>, offset: usize,
                           size: [usize; 2], stride: usize) -> ByteTensor {
        unsafe {
            let tensor =
                THByteTensor_newWithStorage2d(store.borrow_mut().storage(),
                                              offset, size[0], stride,
                                              size[1], 1);
            let dim = THByteTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THByteTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            ByteTensor{forget: false,
                       storage: Some(store),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_3d(store: Rc<RefCell<ByteStorage>>, offset: usize,
                           size: [usize; 3], stride: [usize; 2])
     -> ByteTensor {
        unsafe {
            let tensor =
                THByteTensor_newWithStorage3d(store.borrow_mut().storage(),
                                              offset, size[0], stride[0],
                                              size[1], stride[1], size[2], 1);
            let dim = THByteTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THByteTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            ByteTensor{forget: false,
                       storage: Some(store),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_4d(store: Rc<RefCell<ByteStorage>>, offset: usize,
                           size: [usize; 4], stride: [usize; 3])
     -> ByteTensor {
        unsafe {
            let tensor =
                THByteTensor_newWithStorage4d(store.borrow_mut().storage(),
                                              offset, size[0], stride[0],
                                              size[1], stride[1], size[2],
                                              stride[2], size[3], 1);
            let dim = THByteTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THByteTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            ByteTensor{forget: false,
                       storage: Some(store),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_nd(store: Rc<RefCell<ByteStorage>>, offset: usize,
                           size: &[usize], stride: &[usize]) -> ByteTensor {
        if !(size.len() == stride.len() - 1 || size.len() == stride.len()) {
            {
                ::std::rt::begin_panic("Stride shall have either n - 1 elements or n elements where n = size.len()",
                                       &("tensor\\src\\lib.rs", 1286u32,
                                         1u32))
            }
        };
        let mut storage_len = 0;
        for i in 0..(stride.len() - 1) {
            let cur_len = stride[i] * size[i];
            if cur_len > storage_len { storage_len = cur_len; }
        }
        storage_len += size[size.len() - 1] - 1;
        let mut stride = stride.to_owned();
        if stride.len() == size.len() - 1 { stride.push(1); }
        unsafe {
            let tensor =
                THByteTensor_newWithStorage1d(store.borrow_mut().storage(),
                                              offset, storage_len, 1);
            THByteTensor_resizeNd(tensor, size.len() as i32,
                                  size.as_ptr() as *const i64,
                                  stride.as_ptr() as *const i64);
            let dim = THByteTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THByteTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            let stride = stride.to_vec();
            ByteTensor{forget: false,
                       storage: Some(store),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_1d(size: usize) -> ByteTensor {
        unsafe {
            let tensor = THByteTensor_newWithSize1d(size as i64);
            let stride = THByteTensor_stride(tensor, 0 as c_int) as usize;
            let storage =
                ByteStorage::from(THByteTensor_storage(tensor)).forget();
            ByteTensor{forget: false,
                       storage: Some(Rc::new(RefCell::new(storage))),
                       tensor: tensor,
                       size: <[_]>::into_vec(box [size]),
                       stride: <[_]>::into_vec(box [stride]),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_2d(size: [usize; 2]) -> ByteTensor {
        unsafe {
            let tensor =
                THByteTensor_newWithSize2d(size[0] as i64, size[1] as i64);
            let stride =
                [THByteTensor_stride(tensor, 0 as c_int) as usize,
                 THByteTensor_stride(tensor, 1 as c_int) as usize];
            let storage: ByteStorage = THByteTensor_storage(tensor).into();
            let dim = THByteTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THByteTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            ByteTensor{forget: false,
                       storage: Some(Rc::new(RefCell::new(storage.forget()))),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride.to_vec(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_3d(size: [usize; 3]) -> ByteTensor {
        unsafe {
            let tensor =
                THByteTensor_newWithSize3d(size[0] as i64, size[1] as i64,
                                           size[2] as i64);
            let stride =
                [THByteTensor_stride(tensor, 0 as c_int) as usize,
                 THByteTensor_stride(tensor, 1 as c_int) as usize,
                 THByteTensor_stride(tensor, 2 as c_int) as usize];
            let storage =
                ByteStorage::from(THByteTensor_storage(tensor)).forget();
            let dim = THByteTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THByteTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            ByteTensor{forget: false,
                       storage: Some(Rc::new(RefCell::new(storage))),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride.to_vec(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_4d(size: [usize; 4]) -> ByteTensor {
        unsafe {
            let tensor =
                THByteTensor_newWithSize4d(size[0] as i64, size[1] as i64,
                                           size[2] as i64, size[3] as i64);
            let stride =
                [THByteTensor_stride(tensor, 0 as c_int) as usize,
                 THByteTensor_stride(tensor, 1 as c_int) as usize,
                 THByteTensor_stride(tensor, 2 as c_int) as usize,
                 THByteTensor_stride(tensor, 3 as c_int) as usize];
            let storage =
                ByteStorage::from(THByteTensor_storage(tensor)).forget();
            let dim = THByteTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THByteTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            ByteTensor{forget: false,
                       storage: Some(Rc::new(RefCell::new(storage))),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride.to_vec(),}
        }
    }
}
impl BasicManipulateOp<ByteStorage> for ByteTensor {
    type
    Datum
    =
    u8;
    fn desc(&self) -> String {
        unsafe { THByteTensor_desc(self.tensor).to_string() }
    }
    fn dimensions(&self) -> usize {
        unsafe { THByteTensor_nDimension(self.tensor) as usize }
    }
    fn get_0d(&self) -> u8 { unsafe { THByteTensor_get0d(self.tensor) } }
    fn get_1d(&self, i: usize) -> u8 {
        unsafe { THByteTensor_get1d(self.tensor, i as i64) }
    }
    fn get_2d(&self, i: [usize; 2]) -> u8 {
        unsafe { THByteTensor_get2d(self.tensor, i[0] as i64, i[1] as i64) }
    }
    fn get_3d(&self, i: [usize; 3]) -> u8 {
        unsafe {
            THByteTensor_get3d(self.tensor, i[0] as i64, i[1] as i64,
                               i[2] as i64)
        }
    }
    fn get_4d(&self, i: [usize; 4]) -> u8 {
        unsafe {
            THByteTensor_get4d(self.tensor, i[0] as i64, i[1] as i64,
                               i[2] as i64, i[3] as i64)
        }
    }
    fn iter(&self) -> TensorIterator<u8> { self.into_iter() }
    fn iter_mut(&mut self) -> TensorIterMut<u8> { self.into_iter() }
    fn is_contiguous(&self) -> bool {
        unsafe { THByteTensor_isContiguous(self.tensor) != 0 }
    }
    fn numel(&self) -> usize {
        unsafe { THByteTensor_numel(self.tensor) as usize }
    }
    #[cfg(feature = "safe")]
    fn resize_0d(&mut self) {
        unsafe {
            self.size.clear();
            self.stride.clear();
            THByteTensor_resize0d(self.tensor);
        }
    }
    #[cfg(feature = "safe")]
    fn resize_1d(&mut self, size: usize) {
        unsafe {
            self.size = <[_]>::into_vec(box [size]);
            self.stride = <[_]>::into_vec(box [1]);
            THByteTensor_resize1d(self.tensor, size as i64);
            self.stride =
                <[_]>::into_vec(box
                                    [THByteTensor_stride(self.tensor,
                                                         0 as c_int) as
                                         usize]);
        }
    }
    #[cfg(feature = "safe")]
    fn resize_2d(&mut self, size: [usize; 2]) {
        unsafe {
            self.size = size.to_vec();
            THByteTensor_resize2d(self.tensor, size[0] as i64,
                                  size[1] as i64);
            self.stride =
                [THByteTensor_stride(self.tensor, 0 as c_int) as usize,
                 THByteTensor_stride(self.tensor, 1 as c_int) as
                     usize].to_vec();
        }
    }
    #[cfg(feature = "safe")]
    fn resize_3d(&mut self, size: [usize; 3]) {
        unsafe {
            self.size = size.to_vec();
            THByteTensor_resize3d(self.tensor, size[0] as i64, size[1] as i64,
                                  size[2] as i64);
            self.stride =
                [THByteTensor_stride(self.tensor, 0 as c_int) as usize,
                 THByteTensor_stride(self.tensor, 1 as c_int) as usize,
                 THByteTensor_stride(self.tensor, 2 as c_int) as
                     usize].to_vec();
        }
    }
    #[cfg(feature = "safe")]
    fn resize_nd(&mut self, size: &[usize], stride: &[usize]) {
        {
            match (&(size.len() - 1), &(stride.len())) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        {
                            ::std::rt::begin_panic_fmt(&::std::fmt::Arguments::new_v1(&["assertion failed: `(left == right)`\n  left: `",
                                                                                        "`,\n right: `",
                                                                                        "`: "],
                                                                                      &match (&left_val,
                                                                                              &right_val,
                                                                                              &::std::fmt::Arguments::new_v1(&["Stride must have exactly ",
                                                                                                                               " elements"],
                                                                                                                             &match (&(size.len()
                                                                                                                                           -
                                                                                                                                           1),)
                                                                                                                                  {
                                                                                                                                  (arg0,)
                                                                                                                                  =>
                                                                                                                                  [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                                                               ::std::fmt::Display::fmt)],
                                                                                                                              }))
                                                                                           {
                                                                                           (arg0,
                                                                                            arg1,
                                                                                            arg2)
                                                                                           =>
                                                                                           [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                        ::std::fmt::Debug::fmt),
                                                                                            ::std::fmt::ArgumentV1::new(arg1,
                                                                                                                        ::std::fmt::Debug::fmt),
                                                                                            ::std::fmt::ArgumentV1::new(arg2,
                                                                                                                        ::std::fmt::Display::fmt)],
                                                                                       }),
                                                       &("tensor\\src\\lib.rs",
                                                         1286u32, 1u32))
                        }
                    }
                }
            }
        };
        unsafe {
            self.size = size.to_owned();
            self.stride = stride.to_owned();
            self.stride.push(1);
            THByteTensor_resizeNd(self.tensor, size.len() as c_int,
                                  size.as_ptr() as *const i64,
                                  self.stride.as_ptr() as *const i64);
        }
    }
    fn set_0d(&mut self, v: u8) {
        unsafe { THByteTensor_set0d(self.tensor, v); }
    }
    fn set_1d(&mut self, i: usize, v: u8) {
        unsafe { THByteTensor_set1d(self.tensor, i as i64, v); }
    }
    fn set_2d(&mut self, i: [usize; 2], v: u8) {
        unsafe {
            THByteTensor_set2d(self.tensor, i[0] as i64, i[1] as i64, v);
        }
    }
    fn set_3d(&mut self, i: [usize; 3], v: u8) {
        unsafe {
            THByteTensor_set3d(self.tensor, i[0] as i64, i[1] as i64,
                               i[2] as i64, v);
        }
    }
    fn set_4d(&mut self, i: [usize; 4], v: u8) {
        unsafe {
            THByteTensor_set4d(self.tensor, i[0] as i64, i[1] as i64,
                               i[2] as i64, i[3] as i64, v);
        }
    }
    fn shape(&self) -> (&[usize], &[usize]) {
        (self.size.as_slice(), self.stride.as_slice())
    }
    fn size(&self, dim: usize) -> usize {
        if !(dim < self.size.len()) {
            {
                ::std::rt::begin_panic("assertion failed: dim < self.size.len()",
                                       &("tensor\\src\\lib.rs", 1286u32,
                                         1u32))
            }
        };
        unsafe { THByteTensor_size(self.tensor, dim as c_int) as usize }
    }
    #[cfg(feature = "safe")]
    fn storage(&mut self) -> &mut Option<Rc<RefCell<ByteStorage>>> {
        &mut self.storage
    }
    fn storage_offset(&self) -> usize {
        unsafe { THByteTensor_storageOffset(self.tensor) as usize }
    }
    fn stride(&self, dim: usize) -> usize {
        if !(dim < self.stride.len()) {
            {
                ::std::rt::begin_panic("assertion failed: dim < self.stride.len()",
                                       &("tensor\\src\\lib.rs", 1286u32,
                                         1u32))
            }
        };
        unsafe { THByteTensor_stride(self.tensor, dim as c_int) as usize }
    }
}
impl UtilityOp<ByteStorage> for ByteTensor { }
impl ViewOp<ByteTensor> for ByteTensor {
    #[cfg(feature = "safe")]
    fn narrow(&self, bound: &[Range<usize>])
     -> Result<ByteTensor, NarrowError> {
        let (cur_shape, cur_stride) = self.shape();
        let mut new_size = Vec::with_capacity(bound.len());
        let mut offset = self.storage_offset();
        for (((dim, u_bound), cur_stride), new_bound) in
            cur_shape.iter().enumerate().zip(cur_stride.iter()).zip(bound.iter())
            {
            if *u_bound < new_bound.end {
                return Err(NarrowError{dim: dim,})
            } else {
                new_size.push(new_bound.end - new_bound.start);
                offset += new_bound.start * *cur_stride;
            }
        }
        let storage =
            Rc::clone(self.storage.as_ref().ok_or(NarrowError{dim: 0,})?);
        let tensor =
            ByteTensor::new_with_storage_nd(storage, offset, &new_size,
                                            &cur_stride);
        Ok(tensor)
    }
    #[cfg(feature = "safe")]
    fn narrow_on(&self, dim: usize, new_bound: Range<usize>)
     -> Result<ByteTensor, NarrowError> {
        let (cur_shape, _) = self.shape();
        if new_bound.end <= cur_shape[dim] {
        } else { return Err(NarrowError{dim: dim,}) }
        let tensor =
            self.new_narrow(dim, new_bound.start,
                            new_bound.end - new_bound.start);
        Ok(tensor)
    }
    #[cfg(feature = "safe")]
    fn squeeze(&self) -> ByteTensor {
        let mut new_ts = Self::new();
        new_ts.storage =
            match self.storage {
                Some(ref s) => Some(Rc::clone(s)),
                None => None,
            };
        unsafe {
            THByteTensor_squeeze(new_ts.tensor, self.tensor);
            let dim = THByteTensor_nDimension(new_ts.tensor);
            for i in 0..(dim - 1) {
                new_ts.size.push(THByteTensor_size(new_ts.tensor, i as c_int)
                                     as usize);
                new_ts.stride.push(THByteTensor_stride(new_ts.tensor,
                                                       i as c_int) as usize);
            }
            new_ts.size.push(THByteTensor_size(new_ts.tensor,
                                               dim - 1 as c_int) as usize);
            new_ts.stride.push(THByteTensor_stride(new_ts.tensor,
                                                   dim - 1 as c_int) as
                                   usize);
            new_ts
        }
    }
    #[cfg(feature = "safe")]
    fn view(&self, sizes: &[Option<usize>]) -> Result<ByteTensor, ViewError> {
        let new_size = self.infer_size(sizes)?;
        let new_stride = self.compute_stride(&new_size)?;
        let offset = self.storage_offset();
        let mut storage =
            Rc::clone(self.storage.as_ref().ok_or(SizeInferError::ElementSizeMismatch)?);
        let ts =
            ByteTensor::new_with_storage_nd(storage, offset, &new_size,
                                            &new_stride);
        Ok(ts)
    }
}
impl Tensor for ByteTensor {
    type
    Datum
    =
    u8;
    type
    Storage
    =
    ByteStorage;
}
impl <'a> Clone for ByteTensor {
    #[cfg(feature = "safe")]
    fn clone(&self) -> ByteTensor {
        unsafe {
            let clone = THByteTensor_newClone(self.tensor);
            let storage =
                Some(Rc::new(RefCell::new(ByteStorage::from(THByteTensor_storage(clone)).forget())));
            let stride: Vec<usize> =
                (0..THByteTensor_nDimension(clone)).map(|i|
                                                            {
                                                                THByteTensor_stride(clone,
                                                                                    i
                                                                                        as
                                                                                        i32)
                                                                    as usize
                                                            }).collect();
            ByteTensor{forget: false,
                       storage: storage,
                       tensor: clone,
                       size: self.size.to_owned(),
                       stride: stride,}
        }
    }
}
impl Drop for ByteTensor {
    fn drop(&mut self) {
        unsafe { if !self.forget { THByteTensor_free(self.tensor); } }
    }
}
impl <'a> IntoIterator for &'a ByteTensor {
    type
    Item
    =
    u8;
    type
    IntoIter
    =
    TensorIterator<'a, u8>;
    fn into_iter(self) -> Self::IntoIter {
        let (size, stride) = self.shape();
        TensorIterator::new(self.data(), size, stride)
    }
}
impl <'a> IntoIterator for &'a mut ByteTensor {
    type
    Item
    =
    &'a mut u8;
    type
    IntoIter
    =
    TensorIterMut<'a, u8>;
    fn into_iter(self) -> Self::IntoIter {
        unsafe {
            let data = &mut *(self.data_mut() as *mut [u8]);
            let (size, stride) = self.shape();
            TensorIterMut::new(data, size, stride)
        }
    }
}
impl <'a> From<&'a [u8]> for ByteTensor {
    #[cfg(feature = "safe")]
    fn from(slice: &'a [u8]) -> ByteTensor {
        let mut tensor = ByteTensor::new_with_size_1d(slice.len());
        tensor.storage().borrow_mut().unwrap().iter_mut().enumerate().for_each(|(i,
                                                                                 v)|
                                                                                   *v
                                                                                       =
                                                                                       slice[i]);
        tensor
    }
}
use storage::THCharStorage;
#[repr(C)]
pub struct THCharTensor;
#[link(name = "caffe2")]
extern "C" {
    fn THCharTensor_new() -> *mut THCharTensor;
    fn THCharTensor_newClone(org: *const THCharTensor) -> *mut THCharTensor;
    fn THCharTensor_newContiguous(org: *const THCharTensor)
     -> *mut THCharTensor;
    fn THCharTensor_newNarrow(org: *const THCharTensor, dim: c_int, i: i64,
                              size: i64) -> *mut THCharTensor;
    fn THCharTensor_newSelect(org: *const THCharTensor, dim: c_int, idx: i64)
     -> *mut THCharTensor;
    fn THCharTensor_newTranspose(org: *const THCharTensor, dim_1: c_int,
                                 dim_2: c_int) -> *mut THCharTensor;
    fn THCharTensor_newUnfold(org: *const THCharTensor, dim: c_int, size: i64,
                              step: i64) -> *mut THCharTensor;
    fn THCharTensor_newWithTensor(org: *const THCharTensor)
     -> *mut THCharTensor;
    fn THCharTensor_newWithStorage1d(store: *mut THCharStorage, offset: usize,
                                     size: usize, stride: usize)
     -> *mut THCharTensor;
    fn THCharTensor_newWithStorage2d(store: *mut THCharStorage, offset: usize,
                                     size_1: usize, stride_1: usize,
                                     size_2: usize, stride_2: usize)
     -> *mut THCharTensor;
    fn THCharTensor_newWithStorage3d(store: *mut THCharStorage, offset: usize,
                                     size_1: usize, stride_1: usize,
                                     size_2: usize, stride_2: usize,
                                     size_3: usize, stride_3: usize)
     -> *mut THCharTensor;
    fn THCharTensor_newWithStorage4d(store: *mut THCharStorage, offset: usize,
                                     size_1: usize, stride_1: usize,
                                     size_2: usize, stride_2: usize,
                                     size_3: usize, stride_3: usize,
                                     size_4: usize, stride_4: usize)
     -> *mut THCharTensor;
    fn THCharTensor_newWithSize1d(size: i64) -> *mut THCharTensor;
    fn THCharTensor_newWithSize2d(size_1: i64, size_2: i64)
     -> *mut THCharTensor;
    fn THCharTensor_newWithSize3d(size_1: i64, size_2: i64, size_3: i64)
     -> *mut THCharTensor;
    fn THCharTensor_newWithSize4d(size_1: i64, size_2: i64, size_3: i64,
                                  size_4: i64) -> *mut THCharTensor;
    fn THCharTensor_free(tensor: *mut THCharTensor);
    fn THCharTensor_data(tensor: *mut THCharTensor) -> *mut i8;
    fn THCharTensor_desc(tensor: *mut THCharTensor) -> THDescBuff;
    fn THCharTensor_nDimension(tensor: *const THCharTensor) -> c_int;
    fn THCharTensor_isContiguous(tensor: *const THCharTensor) -> c_int;
    fn THCharTensor_get0d(tensor: *const THCharTensor) -> i8;
    fn THCharTensor_get1d(tensor: *const THCharTensor, i: i64) -> i8;
    fn THCharTensor_get2d(tensor: *const THCharTensor, i: i64, j: i64) -> i8;
    fn THCharTensor_get3d(tensor: *const THCharTensor, i: i64, j: i64, k: i64)
     -> i8;
    fn THCharTensor_get4d(tensor: *const THCharTensor, i: i64, j: i64, k: i64,
                          l: i64) -> i8;
    fn THCharTensor_numel(tensor: *const THCharTensor) -> usize;
    fn THCharTensor_resize0d(tensor: *mut THCharTensor);
    fn THCharTensor_resize1d(tensor: *mut THCharTensor, size_1: i64);
    fn THCharTensor_resize2d(tensor: *mut THCharTensor, size_1: i64,
                             size_2: i64);
    fn THCharTensor_resize3d(tensor: *mut THCharTensor, size_1: i64,
                             size_2: i64, size_3: i64);
    fn THCharTensor_resize4d(tensor: *mut THCharTensor, size_1: i64,
                             size_2: i64, size_3: i64, size_4: i64);
    fn THCharTensor_resize5d(tensor: *mut THCharTensor, size_1: i64,
                             size_2: i64, size_3: i64, size_4: i64,
                             size_5: i64);
    fn THCharTensor_resizeNd(tensor: *mut THCharTensor, dim: c_int,
                             size: *const i64, stride: *const i64);
    fn THCharTensor_set0d(tensor: *const THCharTensor, v: i8);
    fn THCharTensor_set1d(tensor: *const THCharTensor, i: i64, v: i8);
    fn THCharTensor_set2d(tensor: *const THCharTensor, i: i64, j: i64, v: i8);
    fn THCharTensor_set3d(tensor: *const THCharTensor, i: i64, j: i64, k: i64,
                          v: i8);
    fn THCharTensor_set4d(tensor: *const THCharTensor, i: i64, j: i64, k: i64,
                          l: i64, v: i8);
    fn THCharTensor_size(tensor: *const THCharTensor, dim: c_int) -> i64;
    fn THCharTensor_setStorageNd(tensor: *const THCharTensor,
                                 storage: THCharStorage, offset: i64,
                                 dim: c_int, size: *const i64,
                                 stride: *const i64);
    fn THCharTensor_storage(tensor: *mut THCharTensor) -> *mut THCharStorage;
    fn THCharTensor_storageOffset(tensor: *const THCharTensor) -> i64;
    fn THCharTensor_stride(tensor: *const THCharTensor, dim: c_int) -> i64;
    fn THCharTensor_squeeze(tensor: *mut THCharTensor,
                            src: *const THCharTensor);
}
pub struct CharTensor {
    forget: bool,
    #[cfg(feature = "safe")]
    storage: Option<Rc<RefCell<CharStorage>>>,
    tensor: *mut THCharTensor,
    size: Vec<usize>,
    stride: Vec<usize>,
}
impl CharTensor {
    #[doc = r" Get short description of storage."]
    #[doc = r" This includes name of storage, size, and"]
    #[doc = r" sample data if it has more than 20 elements."]
    #[doc = r" If it has less than 20 elements, it'll display"]
    #[doc = r" every elements."]
    fn short_desc(&mut self) -> String {
        #[cfg(feature = "safe")]
        fn get_data(s: &CharTensor) -> &CharStorage {
            s.storage().as_ref().unwrap().borrow()
        }
        let size = self.size.as_slice();
        let stride = self.stride.as_slice();
        let data = get_data(self);
        let name = "CharTensor";
        if size.iter().fold(0, |cum, v| cum + v) > 20 {
            ::alloc::fmt::format(::std::fmt::Arguments::new_v1(&["", ":size=",
                                                                 ":stride=",
                                                                 ":first(10)=",
                                                                 ":last(10)="],
                                                               &match (&name,
                                                                       &size,
                                                                       &stride,
                                                                       &&(&*data)[0..10],
                                                                       &&(&*data)[(data.len()
                                                                                       -
                                                                                       10)..data.len()])
                                                                    {
                                                                    (arg0,
                                                                     arg1,
                                                                     arg2,
                                                                     arg3,
                                                                     arg4) =>
                                                                    [::std::fmt::ArgumentV1::new(arg0,
                                                                                                 ::std::fmt::Display::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg1,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg2,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg3,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg4,
                                                                                                 ::std::fmt::Debug::fmt)],
                                                                }))
        } else {
            ::alloc::fmt::format(::std::fmt::Arguments::new_v1(&["", ":size=",
                                                                 ":stride=",
                                                                 ":data="],
                                                               &match (&name,
                                                                       &size,
                                                                       &stride,
                                                                       &data.iter())
                                                                    {
                                                                    (arg0,
                                                                     arg1,
                                                                     arg2,
                                                                     arg3) =>
                                                                    [::std::fmt::ArgumentV1::new(arg0,
                                                                                                 ::std::fmt::Display::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg1,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg2,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg3,
                                                                                                 ::std::fmt::Debug::fmt)],
                                                                }))
        }
    }
    #[doc = r" Alias for short_desc"]
    #[inline(always)]
    fn to_string(&mut self) -> String { self.short_desc() }
}
impl CreateOp<CharStorage> for CharTensor {
    type
    Datum
    =
    i8;
    #[cfg(feature = "safe")]
    fn new() -> CharTensor {
        unsafe {
            CharTensor{forget: false,
                       storage: None,
                       tensor: THCharTensor_new(),
                       size: Vec::new(),
                       stride: Vec::new(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_contiguous(&self) -> CharTensor {
        unsafe {
            let cont = THCharTensor_newContiguous(self.tensor);
            let storage =
                Some(Rc::new(RefCell::new(CharStorage::from(THCharTensor_storage(cont)))));
            let stride: Vec<usize> =
                (0..THCharTensor_nDimension(cont)).map(|i|
                                                           {
                                                               THCharTensor_stride(cont,
                                                                                   i
                                                                                       as
                                                                                       i32)
                                                                   as usize
                                                           }).collect();
            CharTensor{forget: false,
                       storage: storage,
                       tensor: cont,
                       size: self.size.to_owned(),
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_narrow(&self, dim: usize, i: usize, size: usize) -> CharTensor {
        unsafe {
            let tensor =
                THCharTensor_newNarrow(self.tensor, dim as c_int, i as i64,
                                       size as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THCharTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THCharTensor_stride(tensor, i as i32)
                                             as usize;
                                     let cur_size =
                                         THCharTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            CharTensor{forget: false,
                       storage: storage,
                       tensor: tensor,
                       size: size,
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_select(&self, dim: usize, i: usize) -> CharTensor {
        unsafe {
            let tensor =
                THCharTensor_newSelect(self.tensor, dim as c_int, i as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THCharTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THCharTensor_stride(tensor, i as i32)
                                             as usize;
                                     let cur_size =
                                         THCharTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            CharTensor{forget: false,
                       storage: storage,
                       tensor: tensor,
                       size: size,
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_transpose(&self, dim_1: usize, dim_2: usize) -> CharTensor {
        unsafe {
            let tensor =
                THCharTensor_newTranspose(self.tensor, dim_1 as c_int,
                                          dim_2 as c_int);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THCharTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THCharTensor_stride(tensor, i as i32)
                                             as usize;
                                     let cur_size =
                                         THCharTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            CharTensor{forget: false,
                       storage: storage,
                       tensor: tensor,
                       size: size,
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_unfold(&self, dim: usize, size: usize, step: usize) -> CharTensor {
        unsafe {
            let tensor =
                THCharTensor_newUnfold(self.tensor, dim as c_int, size as i64,
                                       step as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THCharTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THCharTensor_stride(tensor, i as i32)
                                             as usize;
                                     let cur_size =
                                         THCharTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            CharTensor{forget: false,
                       storage: storage,
                       tensor: tensor,
                       size: size,
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_1d(store: Rc<RefCell<CharStorage>>, offset: usize,
                           size: usize) -> CharTensor {
        unsafe {
            let tensor =
                THCharTensor_newWithStorage1d(store.borrow_mut().storage(),
                                              offset, size, 1);
            CharTensor{forget: false,
                       storage: Some(store),
                       tensor: tensor,
                       size: <[_]>::into_vec(box [size]),
                       stride: <[_]>::into_vec(box [1]),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_2d(store: Rc<RefCell<CharStorage>>, offset: usize,
                           size: [usize; 2], stride: usize) -> CharTensor {
        unsafe {
            let tensor =
                THCharTensor_newWithStorage2d(store.borrow_mut().storage(),
                                              offset, size[0], stride,
                                              size[1], 1);
            let dim = THCharTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THCharTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            CharTensor{forget: false,
                       storage: Some(store),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_3d(store: Rc<RefCell<CharStorage>>, offset: usize,
                           size: [usize; 3], stride: [usize; 2])
     -> CharTensor {
        unsafe {
            let tensor =
                THCharTensor_newWithStorage3d(store.borrow_mut().storage(),
                                              offset, size[0], stride[0],
                                              size[1], stride[1], size[2], 1);
            let dim = THCharTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THCharTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            CharTensor{forget: false,
                       storage: Some(store),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_4d(store: Rc<RefCell<CharStorage>>, offset: usize,
                           size: [usize; 4], stride: [usize; 3])
     -> CharTensor {
        unsafe {
            let tensor =
                THCharTensor_newWithStorage4d(store.borrow_mut().storage(),
                                              offset, size[0], stride[0],
                                              size[1], stride[1], size[2],
                                              stride[2], size[3], 1);
            let dim = THCharTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THCharTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            CharTensor{forget: false,
                       storage: Some(store),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_nd(store: Rc<RefCell<CharStorage>>, offset: usize,
                           size: &[usize], stride: &[usize]) -> CharTensor {
        if !(size.len() == stride.len() - 1 || size.len() == stride.len()) {
            {
                ::std::rt::begin_panic("Stride shall have either n - 1 elements or n elements where n = size.len()",
                                       &("tensor\\src\\lib.rs", 1288u32,
                                         1u32))
            }
        };
        let mut storage_len = 0;
        for i in 0..(stride.len() - 1) {
            let cur_len = stride[i] * size[i];
            if cur_len > storage_len { storage_len = cur_len; }
        }
        storage_len += size[size.len() - 1] - 1;
        let mut stride = stride.to_owned();
        if stride.len() == size.len() - 1 { stride.push(1); }
        unsafe {
            let tensor =
                THCharTensor_newWithStorage1d(store.borrow_mut().storage(),
                                              offset, storage_len, 1);
            THCharTensor_resizeNd(tensor, size.len() as i32,
                                  size.as_ptr() as *const i64,
                                  stride.as_ptr() as *const i64);
            let dim = THCharTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THCharTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            let stride = stride.to_vec();
            CharTensor{forget: false,
                       storage: Some(store),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_1d(size: usize) -> CharTensor {
        unsafe {
            let tensor = THCharTensor_newWithSize1d(size as i64);
            let stride = THCharTensor_stride(tensor, 0 as c_int) as usize;
            let storage =
                CharStorage::from(THCharTensor_storage(tensor)).forget();
            CharTensor{forget: false,
                       storage: Some(Rc::new(RefCell::new(storage))),
                       tensor: tensor,
                       size: <[_]>::into_vec(box [size]),
                       stride: <[_]>::into_vec(box [stride]),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_2d(size: [usize; 2]) -> CharTensor {
        unsafe {
            let tensor =
                THCharTensor_newWithSize2d(size[0] as i64, size[1] as i64);
            let stride =
                [THCharTensor_stride(tensor, 0 as c_int) as usize,
                 THCharTensor_stride(tensor, 1 as c_int) as usize];
            let storage: CharStorage = THCharTensor_storage(tensor).into();
            let dim = THCharTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THCharTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            CharTensor{forget: false,
                       storage: Some(Rc::new(RefCell::new(storage.forget()))),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride.to_vec(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_3d(size: [usize; 3]) -> CharTensor {
        unsafe {
            let tensor =
                THCharTensor_newWithSize3d(size[0] as i64, size[1] as i64,
                                           size[2] as i64);
            let stride =
                [THCharTensor_stride(tensor, 0 as c_int) as usize,
                 THCharTensor_stride(tensor, 1 as c_int) as usize,
                 THCharTensor_stride(tensor, 2 as c_int) as usize];
            let storage =
                CharStorage::from(THCharTensor_storage(tensor)).forget();
            let dim = THCharTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THCharTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            CharTensor{forget: false,
                       storage: Some(Rc::new(RefCell::new(storage))),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride.to_vec(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_4d(size: [usize; 4]) -> CharTensor {
        unsafe {
            let tensor =
                THCharTensor_newWithSize4d(size[0] as i64, size[1] as i64,
                                           size[2] as i64, size[3] as i64);
            let stride =
                [THCharTensor_stride(tensor, 0 as c_int) as usize,
                 THCharTensor_stride(tensor, 1 as c_int) as usize,
                 THCharTensor_stride(tensor, 2 as c_int) as usize,
                 THCharTensor_stride(tensor, 3 as c_int) as usize];
            let storage =
                CharStorage::from(THCharTensor_storage(tensor)).forget();
            let dim = THCharTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THCharTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            CharTensor{forget: false,
                       storage: Some(Rc::new(RefCell::new(storage))),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride.to_vec(),}
        }
    }
}
impl BasicManipulateOp<CharStorage> for CharTensor {
    type
    Datum
    =
    i8;
    fn desc(&self) -> String {
        unsafe { THCharTensor_desc(self.tensor).to_string() }
    }
    fn dimensions(&self) -> usize {
        unsafe { THCharTensor_nDimension(self.tensor) as usize }
    }
    fn get_0d(&self) -> i8 { unsafe { THCharTensor_get0d(self.tensor) } }
    fn get_1d(&self, i: usize) -> i8 {
        unsafe { THCharTensor_get1d(self.tensor, i as i64) }
    }
    fn get_2d(&self, i: [usize; 2]) -> i8 {
        unsafe { THCharTensor_get2d(self.tensor, i[0] as i64, i[1] as i64) }
    }
    fn get_3d(&self, i: [usize; 3]) -> i8 {
        unsafe {
            THCharTensor_get3d(self.tensor, i[0] as i64, i[1] as i64,
                               i[2] as i64)
        }
    }
    fn get_4d(&self, i: [usize; 4]) -> i8 {
        unsafe {
            THCharTensor_get4d(self.tensor, i[0] as i64, i[1] as i64,
                               i[2] as i64, i[3] as i64)
        }
    }
    fn iter(&self) -> TensorIterator<i8> { self.into_iter() }
    fn iter_mut(&mut self) -> TensorIterMut<i8> { self.into_iter() }
    fn is_contiguous(&self) -> bool {
        unsafe { THCharTensor_isContiguous(self.tensor) != 0 }
    }
    fn numel(&self) -> usize {
        unsafe { THCharTensor_numel(self.tensor) as usize }
    }
    #[cfg(feature = "safe")]
    fn resize_0d(&mut self) {
        unsafe {
            self.size.clear();
            self.stride.clear();
            THCharTensor_resize0d(self.tensor);
        }
    }
    #[cfg(feature = "safe")]
    fn resize_1d(&mut self, size: usize) {
        unsafe {
            self.size = <[_]>::into_vec(box [size]);
            self.stride = <[_]>::into_vec(box [1]);
            THCharTensor_resize1d(self.tensor, size as i64);
            self.stride =
                <[_]>::into_vec(box
                                    [THCharTensor_stride(self.tensor,
                                                         0 as c_int) as
                                         usize]);
        }
    }
    #[cfg(feature = "safe")]
    fn resize_2d(&mut self, size: [usize; 2]) {
        unsafe {
            self.size = size.to_vec();
            THCharTensor_resize2d(self.tensor, size[0] as i64,
                                  size[1] as i64);
            self.stride =
                [THCharTensor_stride(self.tensor, 0 as c_int) as usize,
                 THCharTensor_stride(self.tensor, 1 as c_int) as
                     usize].to_vec();
        }
    }
    #[cfg(feature = "safe")]
    fn resize_3d(&mut self, size: [usize; 3]) {
        unsafe {
            self.size = size.to_vec();
            THCharTensor_resize3d(self.tensor, size[0] as i64, size[1] as i64,
                                  size[2] as i64);
            self.stride =
                [THCharTensor_stride(self.tensor, 0 as c_int) as usize,
                 THCharTensor_stride(self.tensor, 1 as c_int) as usize,
                 THCharTensor_stride(self.tensor, 2 as c_int) as
                     usize].to_vec();
        }
    }
    #[cfg(feature = "safe")]
    fn resize_nd(&mut self, size: &[usize], stride: &[usize]) {
        {
            match (&(size.len() - 1), &(stride.len())) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        {
                            ::std::rt::begin_panic_fmt(&::std::fmt::Arguments::new_v1(&["assertion failed: `(left == right)`\n  left: `",
                                                                                        "`,\n right: `",
                                                                                        "`: "],
                                                                                      &match (&left_val,
                                                                                              &right_val,
                                                                                              &::std::fmt::Arguments::new_v1(&["Stride must have exactly ",
                                                                                                                               " elements"],
                                                                                                                             &match (&(size.len()
                                                                                                                                           -
                                                                                                                                           1),)
                                                                                                                                  {
                                                                                                                                  (arg0,)
                                                                                                                                  =>
                                                                                                                                  [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                                                               ::std::fmt::Display::fmt)],
                                                                                                                              }))
                                                                                           {
                                                                                           (arg0,
                                                                                            arg1,
                                                                                            arg2)
                                                                                           =>
                                                                                           [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                        ::std::fmt::Debug::fmt),
                                                                                            ::std::fmt::ArgumentV1::new(arg1,
                                                                                                                        ::std::fmt::Debug::fmt),
                                                                                            ::std::fmt::ArgumentV1::new(arg2,
                                                                                                                        ::std::fmt::Display::fmt)],
                                                                                       }),
                                                       &("tensor\\src\\lib.rs",
                                                         1288u32, 1u32))
                        }
                    }
                }
            }
        };
        unsafe {
            self.size = size.to_owned();
            self.stride = stride.to_owned();
            self.stride.push(1);
            THCharTensor_resizeNd(self.tensor, size.len() as c_int,
                                  size.as_ptr() as *const i64,
                                  self.stride.as_ptr() as *const i64);
        }
    }
    fn set_0d(&mut self, v: i8) {
        unsafe { THCharTensor_set0d(self.tensor, v); }
    }
    fn set_1d(&mut self, i: usize, v: i8) {
        unsafe { THCharTensor_set1d(self.tensor, i as i64, v); }
    }
    fn set_2d(&mut self, i: [usize; 2], v: i8) {
        unsafe {
            THCharTensor_set2d(self.tensor, i[0] as i64, i[1] as i64, v);
        }
    }
    fn set_3d(&mut self, i: [usize; 3], v: i8) {
        unsafe {
            THCharTensor_set3d(self.tensor, i[0] as i64, i[1] as i64,
                               i[2] as i64, v);
        }
    }
    fn set_4d(&mut self, i: [usize; 4], v: i8) {
        unsafe {
            THCharTensor_set4d(self.tensor, i[0] as i64, i[1] as i64,
                               i[2] as i64, i[3] as i64, v);
        }
    }
    fn shape(&self) -> (&[usize], &[usize]) {
        (self.size.as_slice(), self.stride.as_slice())
    }
    fn size(&self, dim: usize) -> usize {
        if !(dim < self.size.len()) {
            {
                ::std::rt::begin_panic("assertion failed: dim < self.size.len()",
                                       &("tensor\\src\\lib.rs", 1288u32,
                                         1u32))
            }
        };
        unsafe { THCharTensor_size(self.tensor, dim as c_int) as usize }
    }
    #[cfg(feature = "safe")]
    fn storage(&mut self) -> &mut Option<Rc<RefCell<CharStorage>>> {
        &mut self.storage
    }
    fn storage_offset(&self) -> usize {
        unsafe { THCharTensor_storageOffset(self.tensor) as usize }
    }
    fn stride(&self, dim: usize) -> usize {
        if !(dim < self.stride.len()) {
            {
                ::std::rt::begin_panic("assertion failed: dim < self.stride.len()",
                                       &("tensor\\src\\lib.rs", 1288u32,
                                         1u32))
            }
        };
        unsafe { THCharTensor_stride(self.tensor, dim as c_int) as usize }
    }
}
impl UtilityOp<CharStorage> for CharTensor { }
impl ViewOp<CharTensor> for CharTensor {
    #[cfg(feature = "safe")]
    fn narrow(&self, bound: &[Range<usize>])
     -> Result<CharTensor, NarrowError> {
        let (cur_shape, cur_stride) = self.shape();
        let mut new_size = Vec::with_capacity(bound.len());
        let mut offset = self.storage_offset();
        for (((dim, u_bound), cur_stride), new_bound) in
            cur_shape.iter().enumerate().zip(cur_stride.iter()).zip(bound.iter())
            {
            if *u_bound < new_bound.end {
                return Err(NarrowError{dim: dim,})
            } else {
                new_size.push(new_bound.end - new_bound.start);
                offset += new_bound.start * *cur_stride;
            }
        }
        let storage =
            Rc::clone(self.storage.as_ref().ok_or(NarrowError{dim: 0,})?);
        let tensor =
            CharTensor::new_with_storage_nd(storage, offset, &new_size,
                                            &cur_stride);
        Ok(tensor)
    }
    #[cfg(feature = "safe")]
    fn narrow_on(&self, dim: usize, new_bound: Range<usize>)
     -> Result<CharTensor, NarrowError> {
        let (cur_shape, _) = self.shape();
        if new_bound.end <= cur_shape[dim] {
        } else { return Err(NarrowError{dim: dim,}) }
        let tensor =
            self.new_narrow(dim, new_bound.start,
                            new_bound.end - new_bound.start);
        Ok(tensor)
    }
    #[cfg(feature = "safe")]
    fn squeeze(&self) -> CharTensor {
        let mut new_ts = Self::new();
        new_ts.storage =
            match self.storage {
                Some(ref s) => Some(Rc::clone(s)),
                None => None,
            };
        unsafe {
            THCharTensor_squeeze(new_ts.tensor, self.tensor);
            let dim = THCharTensor_nDimension(new_ts.tensor);
            for i in 0..(dim - 1) {
                new_ts.size.push(THCharTensor_size(new_ts.tensor, i as c_int)
                                     as usize);
                new_ts.stride.push(THCharTensor_stride(new_ts.tensor,
                                                       i as c_int) as usize);
            }
            new_ts.size.push(THCharTensor_size(new_ts.tensor,
                                               dim - 1 as c_int) as usize);
            new_ts.stride.push(THCharTensor_stride(new_ts.tensor,
                                                   dim - 1 as c_int) as
                                   usize);
            new_ts
        }
    }
    #[cfg(feature = "safe")]
    fn view(&self, sizes: &[Option<usize>]) -> Result<CharTensor, ViewError> {
        let new_size = self.infer_size(sizes)?;
        let new_stride = self.compute_stride(&new_size)?;
        let offset = self.storage_offset();
        let mut storage =
            Rc::clone(self.storage.as_ref().ok_or(SizeInferError::ElementSizeMismatch)?);
        let ts =
            CharTensor::new_with_storage_nd(storage, offset, &new_size,
                                            &new_stride);
        Ok(ts)
    }
}
impl Tensor for CharTensor {
    type
    Datum
    =
    i8;
    type
    Storage
    =
    CharStorage;
}
impl <'a> Clone for CharTensor {
    #[cfg(feature = "safe")]
    fn clone(&self) -> CharTensor {
        unsafe {
            let clone = THCharTensor_newClone(self.tensor);
            let storage =
                Some(Rc::new(RefCell::new(CharStorage::from(THCharTensor_storage(clone)).forget())));
            let stride: Vec<usize> =
                (0..THCharTensor_nDimension(clone)).map(|i|
                                                            {
                                                                THCharTensor_stride(clone,
                                                                                    i
                                                                                        as
                                                                                        i32)
                                                                    as usize
                                                            }).collect();
            CharTensor{forget: false,
                       storage: storage,
                       tensor: clone,
                       size: self.size.to_owned(),
                       stride: stride,}
        }
    }
}
impl Drop for CharTensor {
    fn drop(&mut self) {
        unsafe { if !self.forget { THCharTensor_free(self.tensor); } }
    }
}
impl <'a> IntoIterator for &'a CharTensor {
    type
    Item
    =
    i8;
    type
    IntoIter
    =
    TensorIterator<'a, i8>;
    fn into_iter(self) -> Self::IntoIter {
        let (size, stride) = self.shape();
        TensorIterator::new(self.data(), size, stride)
    }
}
impl <'a> IntoIterator for &'a mut CharTensor {
    type
    Item
    =
    &'a mut i8;
    type
    IntoIter
    =
    TensorIterMut<'a, i8>;
    fn into_iter(self) -> Self::IntoIter {
        unsafe {
            let data = &mut *(self.data_mut() as *mut [i8]);
            let (size, stride) = self.shape();
            TensorIterMut::new(data, size, stride)
        }
    }
}
impl <'a> From<&'a [i8]> for CharTensor {
    #[cfg(feature = "safe")]
    fn from(slice: &'a [i8]) -> CharTensor {
        let mut tensor = CharTensor::new_with_size_1d(slice.len());
        tensor.storage().borrow_mut().unwrap().iter_mut().enumerate().for_each(|(i,
                                                                                 v)|
                                                                                   *v
                                                                                       =
                                                                                       slice[i]);
        tensor
    }
}
use storage::THDoubleStorage;
#[repr(C)]
pub struct THDoubleTensor;
#[link(name = "caffe2")]
extern "C" {
    fn THDoubleTensor_new() -> *mut THDoubleTensor;
    fn THDoubleTensor_newClone(org: *const THDoubleTensor)
     -> *mut THDoubleTensor;
    fn THDoubleTensor_newContiguous(org: *const THDoubleTensor)
     -> *mut THDoubleTensor;
    fn THDoubleTensor_newNarrow(org: *const THDoubleTensor, dim: c_int,
                                i: i64, size: i64) -> *mut THDoubleTensor;
    fn THDoubleTensor_newSelect(org: *const THDoubleTensor, dim: c_int,
                                idx: i64) -> *mut THDoubleTensor;
    fn THDoubleTensor_newTranspose(org: *const THDoubleTensor, dim_1: c_int,
                                   dim_2: c_int) -> *mut THDoubleTensor;
    fn THDoubleTensor_newUnfold(org: *const THDoubleTensor, dim: c_int,
                                size: i64, step: i64) -> *mut THDoubleTensor;
    fn THDoubleTensor_newWithTensor(org: *const THDoubleTensor)
     -> *mut THDoubleTensor;
    fn THDoubleTensor_newWithStorage1d(store: *mut THDoubleStorage,
                                       offset: usize, size: usize,
                                       stride: usize) -> *mut THDoubleTensor;
    fn THDoubleTensor_newWithStorage2d(store: *mut THDoubleStorage,
                                       offset: usize, size_1: usize,
                                       stride_1: usize, size_2: usize,
                                       stride_2: usize)
     -> *mut THDoubleTensor;
    fn THDoubleTensor_newWithStorage3d(store: *mut THDoubleStorage,
                                       offset: usize, size_1: usize,
                                       stride_1: usize, size_2: usize,
                                       stride_2: usize, size_3: usize,
                                       stride_3: usize)
     -> *mut THDoubleTensor;
    fn THDoubleTensor_newWithStorage4d(store: *mut THDoubleStorage,
                                       offset: usize, size_1: usize,
                                       stride_1: usize, size_2: usize,
                                       stride_2: usize, size_3: usize,
                                       stride_3: usize, size_4: usize,
                                       stride_4: usize)
     -> *mut THDoubleTensor;
    fn THDoubleTensor_newWithSize1d(size: i64) -> *mut THDoubleTensor;
    fn THDoubleTensor_newWithSize2d(size_1: i64, size_2: i64)
     -> *mut THDoubleTensor;
    fn THDoubleTensor_newWithSize3d(size_1: i64, size_2: i64, size_3: i64)
     -> *mut THDoubleTensor;
    fn THDoubleTensor_newWithSize4d(size_1: i64, size_2: i64, size_3: i64,
                                    size_4: i64) -> *mut THDoubleTensor;
    fn THDoubleTensor_free(tensor: *mut THDoubleTensor);
    fn THDoubleTensor_data(tensor: *mut THDoubleTensor) -> *mut f64;
    fn THDoubleTensor_desc(tensor: *mut THDoubleTensor) -> THDescBuff;
    fn THDoubleTensor_nDimension(tensor: *const THDoubleTensor) -> c_int;
    fn THDoubleTensor_isContiguous(tensor: *const THDoubleTensor) -> c_int;
    fn THDoubleTensor_get0d(tensor: *const THDoubleTensor) -> f64;
    fn THDoubleTensor_get1d(tensor: *const THDoubleTensor, i: i64) -> f64;
    fn THDoubleTensor_get2d(tensor: *const THDoubleTensor, i: i64, j: i64)
     -> f64;
    fn THDoubleTensor_get3d(tensor: *const THDoubleTensor, i: i64, j: i64,
                            k: i64) -> f64;
    fn THDoubleTensor_get4d(tensor: *const THDoubleTensor, i: i64, j: i64,
                            k: i64, l: i64) -> f64;
    fn THDoubleTensor_numel(tensor: *const THDoubleTensor) -> usize;
    fn THDoubleTensor_resize0d(tensor: *mut THDoubleTensor);
    fn THDoubleTensor_resize1d(tensor: *mut THDoubleTensor, size_1: i64);
    fn THDoubleTensor_resize2d(tensor: *mut THDoubleTensor, size_1: i64,
                               size_2: i64);
    fn THDoubleTensor_resize3d(tensor: *mut THDoubleTensor, size_1: i64,
                               size_2: i64, size_3: i64);
    fn THDoubleTensor_resize4d(tensor: *mut THDoubleTensor, size_1: i64,
                               size_2: i64, size_3: i64, size_4: i64);
    fn THDoubleTensor_resize5d(tensor: *mut THDoubleTensor, size_1: i64,
                               size_2: i64, size_3: i64, size_4: i64,
                               size_5: i64);
    fn THDoubleTensor_resizeNd(tensor: *mut THDoubleTensor, dim: c_int,
                               size: *const i64, stride: *const i64);
    fn THDoubleTensor_set0d(tensor: *const THDoubleTensor, v: f64);
    fn THDoubleTensor_set1d(tensor: *const THDoubleTensor, i: i64, v: f64);
    fn THDoubleTensor_set2d(tensor: *const THDoubleTensor, i: i64, j: i64,
                            v: f64);
    fn THDoubleTensor_set3d(tensor: *const THDoubleTensor, i: i64, j: i64,
                            k: i64, v: f64);
    fn THDoubleTensor_set4d(tensor: *const THDoubleTensor, i: i64, j: i64,
                            k: i64, l: i64, v: f64);
    fn THDoubleTensor_size(tensor: *const THDoubleTensor, dim: c_int) -> i64;
    fn THDoubleTensor_setStorageNd(tensor: *const THDoubleTensor,
                                   storage: THDoubleStorage, offset: i64,
                                   dim: c_int, size: *const i64,
                                   stride: *const i64);
    fn THDoubleTensor_storage(tensor: *mut THDoubleTensor)
     -> *mut THDoubleStorage;
    fn THDoubleTensor_storageOffset(tensor: *const THDoubleTensor) -> i64;
    fn THDoubleTensor_stride(tensor: *const THDoubleTensor, dim: c_int)
     -> i64;
    fn THDoubleTensor_squeeze(tensor: *mut THDoubleTensor,
                              src: *const THDoubleTensor);
}
pub struct DoubleTensor {
    forget: bool,
    #[cfg(feature = "safe")]
    storage: Option<Rc<RefCell<DoubleStorage>>>,
    tensor: *mut THDoubleTensor,
    size: Vec<usize>,
    stride: Vec<usize>,
}
impl DoubleTensor {
    #[doc = r" Get short description of storage."]
    #[doc = r" This includes name of storage, size, and"]
    #[doc = r" sample data if it has more than 20 elements."]
    #[doc = r" If it has less than 20 elements, it'll display"]
    #[doc = r" every elements."]
    fn short_desc(&mut self) -> String {
        #[cfg(feature = "safe")]
        fn get_data(s: &DoubleTensor) -> &DoubleStorage {
            s.storage().as_ref().unwrap().borrow()
        }
        let size = self.size.as_slice();
        let stride = self.stride.as_slice();
        let data = get_data(self);
        let name = "DoubleTensor";
        if size.iter().fold(0, |cum, v| cum + v) > 20 {
            ::alloc::fmt::format(::std::fmt::Arguments::new_v1(&["", ":size=",
                                                                 ":stride=",
                                                                 ":first(10)=",
                                                                 ":last(10)="],
                                                               &match (&name,
                                                                       &size,
                                                                       &stride,
                                                                       &&(&*data)[0..10],
                                                                       &&(&*data)[(data.len()
                                                                                       -
                                                                                       10)..data.len()])
                                                                    {
                                                                    (arg0,
                                                                     arg1,
                                                                     arg2,
                                                                     arg3,
                                                                     arg4) =>
                                                                    [::std::fmt::ArgumentV1::new(arg0,
                                                                                                 ::std::fmt::Display::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg1,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg2,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg3,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg4,
                                                                                                 ::std::fmt::Debug::fmt)],
                                                                }))
        } else {
            ::alloc::fmt::format(::std::fmt::Arguments::new_v1(&["", ":size=",
                                                                 ":stride=",
                                                                 ":data="],
                                                               &match (&name,
                                                                       &size,
                                                                       &stride,
                                                                       &data.iter())
                                                                    {
                                                                    (arg0,
                                                                     arg1,
                                                                     arg2,
                                                                     arg3) =>
                                                                    [::std::fmt::ArgumentV1::new(arg0,
                                                                                                 ::std::fmt::Display::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg1,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg2,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg3,
                                                                                                 ::std::fmt::Debug::fmt)],
                                                                }))
        }
    }
    #[doc = r" Alias for short_desc"]
    #[inline(always)]
    fn to_string(&mut self) -> String { self.short_desc() }
}
impl CreateOp<DoubleStorage> for DoubleTensor {
    type
    Datum
    =
    f64;
    #[cfg(feature = "safe")]
    fn new() -> DoubleTensor {
        unsafe {
            DoubleTensor{forget: false,
                         storage: None,
                         tensor: THDoubleTensor_new(),
                         size: Vec::new(),
                         stride: Vec::new(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_contiguous(&self) -> DoubleTensor {
        unsafe {
            let cont = THDoubleTensor_newContiguous(self.tensor);
            let storage =
                Some(Rc::new(RefCell::new(DoubleStorage::from(THDoubleTensor_storage(cont)))));
            let stride: Vec<usize> =
                (0..THDoubleTensor_nDimension(cont)).map(|i|
                                                             {
                                                                 THDoubleTensor_stride(cont,
                                                                                       i
                                                                                           as
                                                                                           i32)
                                                                     as usize
                                                             }).collect();
            DoubleTensor{forget: false,
                         storage: storage,
                         tensor: cont,
                         size: self.size.to_owned(),
                         stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_narrow(&self, dim: usize, i: usize, size: usize) -> DoubleTensor {
        unsafe {
            let tensor =
                THDoubleTensor_newNarrow(self.tensor, dim as c_int, i as i64,
                                         size as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THDoubleTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THDoubleTensor_stride(tensor,
                                                               i as i32) as
                                             usize;
                                     let cur_size =
                                         THDoubleTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            DoubleTensor{forget: false,
                         storage: storage,
                         tensor: tensor,
                         size: size,
                         stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_select(&self, dim: usize, i: usize) -> DoubleTensor {
        unsafe {
            let tensor =
                THDoubleTensor_newSelect(self.tensor, dim as c_int, i as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THDoubleTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THDoubleTensor_stride(tensor,
                                                               i as i32) as
                                             usize;
                                     let cur_size =
                                         THDoubleTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            DoubleTensor{forget: false,
                         storage: storage,
                         tensor: tensor,
                         size: size,
                         stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_transpose(&self, dim_1: usize, dim_2: usize) -> DoubleTensor {
        unsafe {
            let tensor =
                THDoubleTensor_newTranspose(self.tensor, dim_1 as c_int,
                                            dim_2 as c_int);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THDoubleTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THDoubleTensor_stride(tensor,
                                                               i as i32) as
                                             usize;
                                     let cur_size =
                                         THDoubleTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            DoubleTensor{forget: false,
                         storage: storage,
                         tensor: tensor,
                         size: size,
                         stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_unfold(&self, dim: usize, size: usize, step: usize)
     -> DoubleTensor {
        unsafe {
            let tensor =
                THDoubleTensor_newUnfold(self.tensor, dim as c_int,
                                         size as i64, step as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THDoubleTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THDoubleTensor_stride(tensor,
                                                               i as i32) as
                                             usize;
                                     let cur_size =
                                         THDoubleTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            DoubleTensor{forget: false,
                         storage: storage,
                         tensor: tensor,
                         size: size,
                         stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_1d(store: Rc<RefCell<DoubleStorage>>, offset: usize,
                           size: usize) -> DoubleTensor {
        unsafe {
            let tensor =
                THDoubleTensor_newWithStorage1d(store.borrow_mut().storage(),
                                                offset, size, 1);
            DoubleTensor{forget: false,
                         storage: Some(store),
                         tensor: tensor,
                         size: <[_]>::into_vec(box [size]),
                         stride: <[_]>::into_vec(box [1]),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_2d(store: Rc<RefCell<DoubleStorage>>, offset: usize,
                           size: [usize; 2], stride: usize) -> DoubleTensor {
        unsafe {
            let tensor =
                THDoubleTensor_newWithStorage2d(store.borrow_mut().storage(),
                                                offset, size[0], stride,
                                                size[1], 1);
            let dim = THDoubleTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THDoubleTensor_stride(tensor,
                                                               i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            DoubleTensor{forget: false,
                         storage: Some(store),
                         tensor: tensor,
                         size: size.to_vec(),
                         stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_3d(store: Rc<RefCell<DoubleStorage>>, offset: usize,
                           size: [usize; 3], stride: [usize; 2])
     -> DoubleTensor {
        unsafe {
            let tensor =
                THDoubleTensor_newWithStorage3d(store.borrow_mut().storage(),
                                                offset, size[0], stride[0],
                                                size[1], stride[1], size[2],
                                                1);
            let dim = THDoubleTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THDoubleTensor_stride(tensor,
                                                               i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            DoubleTensor{forget: false,
                         storage: Some(store),
                         tensor: tensor,
                         size: size.to_vec(),
                         stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_4d(store: Rc<RefCell<DoubleStorage>>, offset: usize,
                           size: [usize; 4], stride: [usize; 3])
     -> DoubleTensor {
        unsafe {
            let tensor =
                THDoubleTensor_newWithStorage4d(store.borrow_mut().storage(),
                                                offset, size[0], stride[0],
                                                size[1], stride[1], size[2],
                                                stride[2], size[3], 1);
            let dim = THDoubleTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THDoubleTensor_stride(tensor,
                                                               i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            DoubleTensor{forget: false,
                         storage: Some(store),
                         tensor: tensor,
                         size: size.to_vec(),
                         stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_nd(store: Rc<RefCell<DoubleStorage>>, offset: usize,
                           size: &[usize], stride: &[usize]) -> DoubleTensor {
        if !(size.len() == stride.len() - 1 || size.len() == stride.len()) {
            {
                ::std::rt::begin_panic("Stride shall have either n - 1 elements or n elements where n = size.len()",
                                       &("tensor\\src\\lib.rs", 1291u32,
                                         1u32))
            }
        };
        let mut storage_len = 0;
        for i in 0..(stride.len() - 1) {
            let cur_len = stride[i] * size[i];
            if cur_len > storage_len { storage_len = cur_len; }
        }
        storage_len += size[size.len() - 1] - 1;
        let mut stride = stride.to_owned();
        if stride.len() == size.len() - 1 { stride.push(1); }
        unsafe {
            let tensor =
                THDoubleTensor_newWithStorage1d(store.borrow_mut().storage(),
                                                offset, storage_len, 1);
            THDoubleTensor_resizeNd(tensor, size.len() as i32,
                                    size.as_ptr() as *const i64,
                                    stride.as_ptr() as *const i64);
            let dim = THDoubleTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THDoubleTensor_stride(tensor,
                                                               i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            let stride = stride.to_vec();
            DoubleTensor{forget: false,
                         storage: Some(store),
                         tensor: tensor,
                         size: size.to_vec(),
                         stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_1d(size: usize) -> DoubleTensor {
        unsafe {
            let tensor = THDoubleTensor_newWithSize1d(size as i64);
            let stride = THDoubleTensor_stride(tensor, 0 as c_int) as usize;
            let storage =
                DoubleStorage::from(THDoubleTensor_storage(tensor)).forget();
            DoubleTensor{forget: false,
                         storage: Some(Rc::new(RefCell::new(storage))),
                         tensor: tensor,
                         size: <[_]>::into_vec(box [size]),
                         stride: <[_]>::into_vec(box [stride]),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_2d(size: [usize; 2]) -> DoubleTensor {
        unsafe {
            let tensor =
                THDoubleTensor_newWithSize2d(size[0] as i64, size[1] as i64);
            let stride =
                [THDoubleTensor_stride(tensor, 0 as c_int) as usize,
                 THDoubleTensor_stride(tensor, 1 as c_int) as usize];
            let storage: DoubleStorage =
                THDoubleTensor_storage(tensor).into();
            let dim = THDoubleTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THDoubleTensor_stride(tensor,
                                                               i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            DoubleTensor{forget: false,
                         storage:
                             Some(Rc::new(RefCell::new(storage.forget()))),
                         tensor: tensor,
                         size: size.to_vec(),
                         stride: stride.to_vec(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_3d(size: [usize; 3]) -> DoubleTensor {
        unsafe {
            let tensor =
                THDoubleTensor_newWithSize3d(size[0] as i64, size[1] as i64,
                                             size[2] as i64);
            let stride =
                [THDoubleTensor_stride(tensor, 0 as c_int) as usize,
                 THDoubleTensor_stride(tensor, 1 as c_int) as usize,
                 THDoubleTensor_stride(tensor, 2 as c_int) as usize];
            let storage =
                DoubleStorage::from(THDoubleTensor_storage(tensor)).forget();
            let dim = THDoubleTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THDoubleTensor_stride(tensor,
                                                               i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            DoubleTensor{forget: false,
                         storage: Some(Rc::new(RefCell::new(storage))),
                         tensor: tensor,
                         size: size.to_vec(),
                         stride: stride.to_vec(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_4d(size: [usize; 4]) -> DoubleTensor {
        unsafe {
            let tensor =
                THDoubleTensor_newWithSize4d(size[0] as i64, size[1] as i64,
                                             size[2] as i64, size[3] as i64);
            let stride =
                [THDoubleTensor_stride(tensor, 0 as c_int) as usize,
                 THDoubleTensor_stride(tensor, 1 as c_int) as usize,
                 THDoubleTensor_stride(tensor, 2 as c_int) as usize,
                 THDoubleTensor_stride(tensor, 3 as c_int) as usize];
            let storage =
                DoubleStorage::from(THDoubleTensor_storage(tensor)).forget();
            let dim = THDoubleTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THDoubleTensor_stride(tensor,
                                                               i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            DoubleTensor{forget: false,
                         storage: Some(Rc::new(RefCell::new(storage))),
                         tensor: tensor,
                         size: size.to_vec(),
                         stride: stride.to_vec(),}
        }
    }
}
impl BasicManipulateOp<DoubleStorage> for DoubleTensor {
    type
    Datum
    =
    f64;
    fn desc(&self) -> String {
        unsafe { THDoubleTensor_desc(self.tensor).to_string() }
    }
    fn dimensions(&self) -> usize {
        unsafe { THDoubleTensor_nDimension(self.tensor) as usize }
    }
    fn get_0d(&self) -> f64 { unsafe { THDoubleTensor_get0d(self.tensor) } }
    fn get_1d(&self, i: usize) -> f64 {
        unsafe { THDoubleTensor_get1d(self.tensor, i as i64) }
    }
    fn get_2d(&self, i: [usize; 2]) -> f64 {
        unsafe { THDoubleTensor_get2d(self.tensor, i[0] as i64, i[1] as i64) }
    }
    fn get_3d(&self, i: [usize; 3]) -> f64 {
        unsafe {
            THDoubleTensor_get3d(self.tensor, i[0] as i64, i[1] as i64,
                                 i[2] as i64)
        }
    }
    fn get_4d(&self, i: [usize; 4]) -> f64 {
        unsafe {
            THDoubleTensor_get4d(self.tensor, i[0] as i64, i[1] as i64,
                                 i[2] as i64, i[3] as i64)
        }
    }
    fn iter(&self) -> TensorIterator<f64> { self.into_iter() }
    fn iter_mut(&mut self) -> TensorIterMut<f64> { self.into_iter() }
    fn is_contiguous(&self) -> bool {
        unsafe { THDoubleTensor_isContiguous(self.tensor) != 0 }
    }
    fn numel(&self) -> usize {
        unsafe { THDoubleTensor_numel(self.tensor) as usize }
    }
    #[cfg(feature = "safe")]
    fn resize_0d(&mut self) {
        unsafe {
            self.size.clear();
            self.stride.clear();
            THDoubleTensor_resize0d(self.tensor);
        }
    }
    #[cfg(feature = "safe")]
    fn resize_1d(&mut self, size: usize) {
        unsafe {
            self.size = <[_]>::into_vec(box [size]);
            self.stride = <[_]>::into_vec(box [1]);
            THDoubleTensor_resize1d(self.tensor, size as i64);
            self.stride =
                <[_]>::into_vec(box
                                    [THDoubleTensor_stride(self.tensor,
                                                           0 as c_int) as
                                         usize]);
        }
    }
    #[cfg(feature = "safe")]
    fn resize_2d(&mut self, size: [usize; 2]) {
        unsafe {
            self.size = size.to_vec();
            THDoubleTensor_resize2d(self.tensor, size[0] as i64,
                                    size[1] as i64);
            self.stride =
                [THDoubleTensor_stride(self.tensor, 0 as c_int) as usize,
                 THDoubleTensor_stride(self.tensor, 1 as c_int) as
                     usize].to_vec();
        }
    }
    #[cfg(feature = "safe")]
    fn resize_3d(&mut self, size: [usize; 3]) {
        unsafe {
            self.size = size.to_vec();
            THDoubleTensor_resize3d(self.tensor, size[0] as i64,
                                    size[1] as i64, size[2] as i64);
            self.stride =
                [THDoubleTensor_stride(self.tensor, 0 as c_int) as usize,
                 THDoubleTensor_stride(self.tensor, 1 as c_int) as usize,
                 THDoubleTensor_stride(self.tensor, 2 as c_int) as
                     usize].to_vec();
        }
    }
    #[cfg(feature = "safe")]
    fn resize_nd(&mut self, size: &[usize], stride: &[usize]) {
        {
            match (&(size.len() - 1), &(stride.len())) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        {
                            ::std::rt::begin_panic_fmt(&::std::fmt::Arguments::new_v1(&["assertion failed: `(left == right)`\n  left: `",
                                                                                        "`,\n right: `",
                                                                                        "`: "],
                                                                                      &match (&left_val,
                                                                                              &right_val,
                                                                                              &::std::fmt::Arguments::new_v1(&["Stride must have exactly ",
                                                                                                                               " elements"],
                                                                                                                             &match (&(size.len()
                                                                                                                                           -
                                                                                                                                           1),)
                                                                                                                                  {
                                                                                                                                  (arg0,)
                                                                                                                                  =>
                                                                                                                                  [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                                                               ::std::fmt::Display::fmt)],
                                                                                                                              }))
                                                                                           {
                                                                                           (arg0,
                                                                                            arg1,
                                                                                            arg2)
                                                                                           =>
                                                                                           [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                        ::std::fmt::Debug::fmt),
                                                                                            ::std::fmt::ArgumentV1::new(arg1,
                                                                                                                        ::std::fmt::Debug::fmt),
                                                                                            ::std::fmt::ArgumentV1::new(arg2,
                                                                                                                        ::std::fmt::Display::fmt)],
                                                                                       }),
                                                       &("tensor\\src\\lib.rs",
                                                         1291u32, 1u32))
                        }
                    }
                }
            }
        };
        unsafe {
            self.size = size.to_owned();
            self.stride = stride.to_owned();
            self.stride.push(1);
            THDoubleTensor_resizeNd(self.tensor, size.len() as c_int,
                                    size.as_ptr() as *const i64,
                                    self.stride.as_ptr() as *const i64);
        }
    }
    fn set_0d(&mut self, v: f64) {
        unsafe { THDoubleTensor_set0d(self.tensor, v); }
    }
    fn set_1d(&mut self, i: usize, v: f64) {
        unsafe { THDoubleTensor_set1d(self.tensor, i as i64, v); }
    }
    fn set_2d(&mut self, i: [usize; 2], v: f64) {
        unsafe {
            THDoubleTensor_set2d(self.tensor, i[0] as i64, i[1] as i64, v);
        }
    }
    fn set_3d(&mut self, i: [usize; 3], v: f64) {
        unsafe {
            THDoubleTensor_set3d(self.tensor, i[0] as i64, i[1] as i64,
                                 i[2] as i64, v);
        }
    }
    fn set_4d(&mut self, i: [usize; 4], v: f64) {
        unsafe {
            THDoubleTensor_set4d(self.tensor, i[0] as i64, i[1] as i64,
                                 i[2] as i64, i[3] as i64, v);
        }
    }
    fn shape(&self) -> (&[usize], &[usize]) {
        (self.size.as_slice(), self.stride.as_slice())
    }
    fn size(&self, dim: usize) -> usize {
        if !(dim < self.size.len()) {
            {
                ::std::rt::begin_panic("assertion failed: dim < self.size.len()",
                                       &("tensor\\src\\lib.rs", 1291u32,
                                         1u32))
            }
        };
        unsafe { THDoubleTensor_size(self.tensor, dim as c_int) as usize }
    }
    #[cfg(feature = "safe")]
    fn storage(&mut self) -> &mut Option<Rc<RefCell<DoubleStorage>>> {
        &mut self.storage
    }
    fn storage_offset(&self) -> usize {
        unsafe { THDoubleTensor_storageOffset(self.tensor) as usize }
    }
    fn stride(&self, dim: usize) -> usize {
        if !(dim < self.stride.len()) {
            {
                ::std::rt::begin_panic("assertion failed: dim < self.stride.len()",
                                       &("tensor\\src\\lib.rs", 1291u32,
                                         1u32))
            }
        };
        unsafe { THDoubleTensor_stride(self.tensor, dim as c_int) as usize }
    }
}
impl UtilityOp<DoubleStorage> for DoubleTensor { }
impl ViewOp<DoubleTensor> for DoubleTensor {
    #[cfg(feature = "safe")]
    fn narrow(&self, bound: &[Range<usize>])
     -> Result<DoubleTensor, NarrowError> {
        let (cur_shape, cur_stride) = self.shape();
        let mut new_size = Vec::with_capacity(bound.len());
        let mut offset = self.storage_offset();
        for (((dim, u_bound), cur_stride), new_bound) in
            cur_shape.iter().enumerate().zip(cur_stride.iter()).zip(bound.iter())
            {
            if *u_bound < new_bound.end {
                return Err(NarrowError{dim: dim,})
            } else {
                new_size.push(new_bound.end - new_bound.start);
                offset += new_bound.start * *cur_stride;
            }
        }
        let storage =
            Rc::clone(self.storage.as_ref().ok_or(NarrowError{dim: 0,})?);
        let tensor =
            DoubleTensor::new_with_storage_nd(storage, offset, &new_size,
                                              &cur_stride);
        Ok(tensor)
    }
    #[cfg(feature = "safe")]
    fn narrow_on(&self, dim: usize, new_bound: Range<usize>)
     -> Result<DoubleTensor, NarrowError> {
        let (cur_shape, _) = self.shape();
        if new_bound.end <= cur_shape[dim] {
        } else { return Err(NarrowError{dim: dim,}) }
        let tensor =
            self.new_narrow(dim, new_bound.start,
                            new_bound.end - new_bound.start);
        Ok(tensor)
    }
    #[cfg(feature = "safe")]
    fn squeeze(&self) -> DoubleTensor {
        let mut new_ts = Self::new();
        new_ts.storage =
            match self.storage {
                Some(ref s) => Some(Rc::clone(s)),
                None => None,
            };
        unsafe {
            THDoubleTensor_squeeze(new_ts.tensor, self.tensor);
            let dim = THDoubleTensor_nDimension(new_ts.tensor);
            for i in 0..(dim - 1) {
                new_ts.size.push(THDoubleTensor_size(new_ts.tensor,
                                                     i as c_int) as usize);
                new_ts.stride.push(THDoubleTensor_stride(new_ts.tensor,
                                                         i as c_int) as
                                       usize);
            }
            new_ts.size.push(THDoubleTensor_size(new_ts.tensor,
                                                 dim - 1 as c_int) as usize);
            new_ts.stride.push(THDoubleTensor_stride(new_ts.tensor,
                                                     dim - 1 as c_int) as
                                   usize);
            new_ts
        }
    }
    #[cfg(feature = "safe")]
    fn view(&self, sizes: &[Option<usize>])
     -> Result<DoubleTensor, ViewError> {
        let new_size = self.infer_size(sizes)?;
        let new_stride = self.compute_stride(&new_size)?;
        let offset = self.storage_offset();
        let mut storage =
            Rc::clone(self.storage.as_ref().ok_or(SizeInferError::ElementSizeMismatch)?);
        let ts =
            DoubleTensor::new_with_storage_nd(storage, offset, &new_size,
                                              &new_stride);
        Ok(ts)
    }
}
impl Tensor for DoubleTensor {
    type
    Datum
    =
    f64;
    type
    Storage
    =
    DoubleStorage;
}
impl <'a> Clone for DoubleTensor {
    #[cfg(feature = "safe")]
    fn clone(&self) -> DoubleTensor {
        unsafe {
            let clone = THDoubleTensor_newClone(self.tensor);
            let storage =
                Some(Rc::new(RefCell::new(DoubleStorage::from(THDoubleTensor_storage(clone)).forget())));
            let stride: Vec<usize> =
                (0..THDoubleTensor_nDimension(clone)).map(|i|
                                                              {
                                                                  THDoubleTensor_stride(clone,
                                                                                        i
                                                                                            as
                                                                                            i32)
                                                                      as usize
                                                              }).collect();
            DoubleTensor{forget: false,
                         storage: storage,
                         tensor: clone,
                         size: self.size.to_owned(),
                         stride: stride,}
        }
    }
}
impl Drop for DoubleTensor {
    fn drop(&mut self) {
        unsafe { if !self.forget { THDoubleTensor_free(self.tensor); } }
    }
}
impl <'a> IntoIterator for &'a DoubleTensor {
    type
    Item
    =
    f64;
    type
    IntoIter
    =
    TensorIterator<'a, f64>;
    fn into_iter(self) -> Self::IntoIter {
        let (size, stride) = self.shape();
        TensorIterator::new(self.data(), size, stride)
    }
}
impl <'a> IntoIterator for &'a mut DoubleTensor {
    type
    Item
    =
    &'a mut f64;
    type
    IntoIter
    =
    TensorIterMut<'a, f64>;
    fn into_iter(self) -> Self::IntoIter {
        unsafe {
            let data = &mut *(self.data_mut() as *mut [f64]);
            let (size, stride) = self.shape();
            TensorIterMut::new(data, size, stride)
        }
    }
}
impl <'a> From<&'a [f64]> for DoubleTensor {
    #[cfg(feature = "safe")]
    fn from(slice: &'a [f64]) -> DoubleTensor {
        let mut tensor = DoubleTensor::new_with_size_1d(slice.len());
        tensor.storage().borrow_mut().unwrap().iter_mut().enumerate().for_each(|(i,
                                                                                 v)|
                                                                                   *v
                                                                                       =
                                                                                       slice[i]);
        tensor
    }
}
use storage::THFloatStorage;
#[repr(C)]
pub struct THFloatTensor;
#[link(name = "caffe2")]
extern "C" {
    fn THFloatTensor_new() -> *mut THFloatTensor;
    fn THFloatTensor_newClone(org: *const THFloatTensor)
     -> *mut THFloatTensor;
    fn THFloatTensor_newContiguous(org: *const THFloatTensor)
     -> *mut THFloatTensor;
    fn THFloatTensor_newNarrow(org: *const THFloatTensor, dim: c_int, i: i64,
                               size: i64) -> *mut THFloatTensor;
    fn THFloatTensor_newSelect(org: *const THFloatTensor, dim: c_int,
                               idx: i64) -> *mut THFloatTensor;
    fn THFloatTensor_newTranspose(org: *const THFloatTensor, dim_1: c_int,
                                  dim_2: c_int) -> *mut THFloatTensor;
    fn THFloatTensor_newUnfold(org: *const THFloatTensor, dim: c_int,
                               size: i64, step: i64) -> *mut THFloatTensor;
    fn THFloatTensor_newWithTensor(org: *const THFloatTensor)
     -> *mut THFloatTensor;
    fn THFloatTensor_newWithStorage1d(store: *mut THFloatStorage,
                                      offset: usize, size: usize,
                                      stride: usize) -> *mut THFloatTensor;
    fn THFloatTensor_newWithStorage2d(store: *mut THFloatStorage,
                                      offset: usize, size_1: usize,
                                      stride_1: usize, size_2: usize,
                                      stride_2: usize) -> *mut THFloatTensor;
    fn THFloatTensor_newWithStorage3d(store: *mut THFloatStorage,
                                      offset: usize, size_1: usize,
                                      stride_1: usize, size_2: usize,
                                      stride_2: usize, size_3: usize,
                                      stride_3: usize) -> *mut THFloatTensor;
    fn THFloatTensor_newWithStorage4d(store: *mut THFloatStorage,
                                      offset: usize, size_1: usize,
                                      stride_1: usize, size_2: usize,
                                      stride_2: usize, size_3: usize,
                                      stride_3: usize, size_4: usize,
                                      stride_4: usize) -> *mut THFloatTensor;
    fn THFloatTensor_newWithSize1d(size: i64) -> *mut THFloatTensor;
    fn THFloatTensor_newWithSize2d(size_1: i64, size_2: i64)
     -> *mut THFloatTensor;
    fn THFloatTensor_newWithSize3d(size_1: i64, size_2: i64, size_3: i64)
     -> *mut THFloatTensor;
    fn THFloatTensor_newWithSize4d(size_1: i64, size_2: i64, size_3: i64,
                                   size_4: i64) -> *mut THFloatTensor;
    fn THFloatTensor_free(tensor: *mut THFloatTensor);
    fn THFloatTensor_data(tensor: *mut THFloatTensor) -> *mut f32;
    fn THFloatTensor_desc(tensor: *mut THFloatTensor) -> THDescBuff;
    fn THFloatTensor_nDimension(tensor: *const THFloatTensor) -> c_int;
    fn THFloatTensor_isContiguous(tensor: *const THFloatTensor) -> c_int;
    fn THFloatTensor_get0d(tensor: *const THFloatTensor) -> f32;
    fn THFloatTensor_get1d(tensor: *const THFloatTensor, i: i64) -> f32;
    fn THFloatTensor_get2d(tensor: *const THFloatTensor, i: i64, j: i64)
     -> f32;
    fn THFloatTensor_get3d(tensor: *const THFloatTensor, i: i64, j: i64,
                           k: i64) -> f32;
    fn THFloatTensor_get4d(tensor: *const THFloatTensor, i: i64, j: i64,
                           k: i64, l: i64) -> f32;
    fn THFloatTensor_numel(tensor: *const THFloatTensor) -> usize;
    fn THFloatTensor_resize0d(tensor: *mut THFloatTensor);
    fn THFloatTensor_resize1d(tensor: *mut THFloatTensor, size_1: i64);
    fn THFloatTensor_resize2d(tensor: *mut THFloatTensor, size_1: i64,
                              size_2: i64);
    fn THFloatTensor_resize3d(tensor: *mut THFloatTensor, size_1: i64,
                              size_2: i64, size_3: i64);
    fn THFloatTensor_resize4d(tensor: *mut THFloatTensor, size_1: i64,
                              size_2: i64, size_3: i64, size_4: i64);
    fn THFloatTensor_resize5d(tensor: *mut THFloatTensor, size_1: i64,
                              size_2: i64, size_3: i64, size_4: i64,
                              size_5: i64);
    fn THFloatTensor_resizeNd(tensor: *mut THFloatTensor, dim: c_int,
                              size: *const i64, stride: *const i64);
    fn THFloatTensor_set0d(tensor: *const THFloatTensor, v: f32);
    fn THFloatTensor_set1d(tensor: *const THFloatTensor, i: i64, v: f32);
    fn THFloatTensor_set2d(tensor: *const THFloatTensor, i: i64, j: i64,
                           v: f32);
    fn THFloatTensor_set3d(tensor: *const THFloatTensor, i: i64, j: i64,
                           k: i64, v: f32);
    fn THFloatTensor_set4d(tensor: *const THFloatTensor, i: i64, j: i64,
                           k: i64, l: i64, v: f32);
    fn THFloatTensor_size(tensor: *const THFloatTensor, dim: c_int) -> i64;
    fn THFloatTensor_setStorageNd(tensor: *const THFloatTensor,
                                  storage: THFloatStorage, offset: i64,
                                  dim: c_int, size: *const i64,
                                  stride: *const i64);
    fn THFloatTensor_storage(tensor: *mut THFloatTensor)
     -> *mut THFloatStorage;
    fn THFloatTensor_storageOffset(tensor: *const THFloatTensor) -> i64;
    fn THFloatTensor_stride(tensor: *const THFloatTensor, dim: c_int) -> i64;
    fn THFloatTensor_squeeze(tensor: *mut THFloatTensor,
                             src: *const THFloatTensor);
}
pub struct FloatTensor {
    forget: bool,
    #[cfg(feature = "safe")]
    storage: Option<Rc<RefCell<FloatStorage>>>,
    tensor: *mut THFloatTensor,
    size: Vec<usize>,
    stride: Vec<usize>,
}
impl FloatTensor {
    #[doc = r" Get short description of storage."]
    #[doc = r" This includes name of storage, size, and"]
    #[doc = r" sample data if it has more than 20 elements."]
    #[doc = r" If it has less than 20 elements, it'll display"]
    #[doc = r" every elements."]
    fn short_desc(&mut self) -> String {
        #[cfg(feature = "safe")]
        fn get_data(s: &FloatTensor) -> &FloatStorage {
            s.storage().as_ref().unwrap().borrow()
        }
        let size = self.size.as_slice();
        let stride = self.stride.as_slice();
        let data = get_data(self);
        let name = "FloatTensor";
        if size.iter().fold(0, |cum, v| cum + v) > 20 {
            ::alloc::fmt::format(::std::fmt::Arguments::new_v1(&["", ":size=",
                                                                 ":stride=",
                                                                 ":first(10)=",
                                                                 ":last(10)="],
                                                               &match (&name,
                                                                       &size,
                                                                       &stride,
                                                                       &&(&*data)[0..10],
                                                                       &&(&*data)[(data.len()
                                                                                       -
                                                                                       10)..data.len()])
                                                                    {
                                                                    (arg0,
                                                                     arg1,
                                                                     arg2,
                                                                     arg3,
                                                                     arg4) =>
                                                                    [::std::fmt::ArgumentV1::new(arg0,
                                                                                                 ::std::fmt::Display::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg1,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg2,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg3,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg4,
                                                                                                 ::std::fmt::Debug::fmt)],
                                                                }))
        } else {
            ::alloc::fmt::format(::std::fmt::Arguments::new_v1(&["", ":size=",
                                                                 ":stride=",
                                                                 ":data="],
                                                               &match (&name,
                                                                       &size,
                                                                       &stride,
                                                                       &data.iter())
                                                                    {
                                                                    (arg0,
                                                                     arg1,
                                                                     arg2,
                                                                     arg3) =>
                                                                    [::std::fmt::ArgumentV1::new(arg0,
                                                                                                 ::std::fmt::Display::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg1,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg2,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg3,
                                                                                                 ::std::fmt::Debug::fmt)],
                                                                }))
        }
    }
    #[doc = r" Alias for short_desc"]
    #[inline(always)]
    fn to_string(&mut self) -> String { self.short_desc() }
}
impl CreateOp<FloatStorage> for FloatTensor {
    type
    Datum
    =
    f32;
    #[cfg(feature = "safe")]
    fn new() -> FloatTensor {
        unsafe {
            FloatTensor{forget: false,
                        storage: None,
                        tensor: THFloatTensor_new(),
                        size: Vec::new(),
                        stride: Vec::new(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_contiguous(&self) -> FloatTensor {
        unsafe {
            let cont = THFloatTensor_newContiguous(self.tensor);
            let storage =
                Some(Rc::new(RefCell::new(FloatStorage::from(THFloatTensor_storage(cont)))));
            let stride: Vec<usize> =
                (0..THFloatTensor_nDimension(cont)).map(|i|
                                                            {
                                                                THFloatTensor_stride(cont,
                                                                                     i
                                                                                         as
                                                                                         i32)
                                                                    as usize
                                                            }).collect();
            FloatTensor{forget: false,
                        storage: storage,
                        tensor: cont,
                        size: self.size.to_owned(),
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_narrow(&self, dim: usize, i: usize, size: usize) -> FloatTensor {
        unsafe {
            let tensor =
                THFloatTensor_newNarrow(self.tensor, dim as c_int, i as i64,
                                        size as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THFloatTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THFloatTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     let cur_size =
                                         THFloatTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            FloatTensor{forget: false,
                        storage: storage,
                        tensor: tensor,
                        size: size,
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_select(&self, dim: usize, i: usize) -> FloatTensor {
        unsafe {
            let tensor =
                THFloatTensor_newSelect(self.tensor, dim as c_int, i as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THFloatTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THFloatTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     let cur_size =
                                         THFloatTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            FloatTensor{forget: false,
                        storage: storage,
                        tensor: tensor,
                        size: size,
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_transpose(&self, dim_1: usize, dim_2: usize) -> FloatTensor {
        unsafe {
            let tensor =
                THFloatTensor_newTranspose(self.tensor, dim_1 as c_int,
                                           dim_2 as c_int);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THFloatTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THFloatTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     let cur_size =
                                         THFloatTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            FloatTensor{forget: false,
                        storage: storage,
                        tensor: tensor,
                        size: size,
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_unfold(&self, dim: usize, size: usize, step: usize)
     -> FloatTensor {
        unsafe {
            let tensor =
                THFloatTensor_newUnfold(self.tensor, dim as c_int,
                                        size as i64, step as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THFloatTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THFloatTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     let cur_size =
                                         THFloatTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            FloatTensor{forget: false,
                        storage: storage,
                        tensor: tensor,
                        size: size,
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_1d(store: Rc<RefCell<FloatStorage>>, offset: usize,
                           size: usize) -> FloatTensor {
        unsafe {
            let tensor =
                THFloatTensor_newWithStorage1d(store.borrow_mut().storage(),
                                               offset, size, 1);
            FloatTensor{forget: false,
                        storage: Some(store),
                        tensor: tensor,
                        size: <[_]>::into_vec(box [size]),
                        stride: <[_]>::into_vec(box [1]),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_2d(store: Rc<RefCell<FloatStorage>>, offset: usize,
                           size: [usize; 2], stride: usize) -> FloatTensor {
        unsafe {
            let tensor =
                THFloatTensor_newWithStorage2d(store.borrow_mut().storage(),
                                               offset, size[0], stride,
                                               size[1], 1);
            let dim = THFloatTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THFloatTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            FloatTensor{forget: false,
                        storage: Some(store),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_3d(store: Rc<RefCell<FloatStorage>>, offset: usize,
                           size: [usize; 3], stride: [usize; 2])
     -> FloatTensor {
        unsafe {
            let tensor =
                THFloatTensor_newWithStorage3d(store.borrow_mut().storage(),
                                               offset, size[0], stride[0],
                                               size[1], stride[1], size[2],
                                               1);
            let dim = THFloatTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THFloatTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            FloatTensor{forget: false,
                        storage: Some(store),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_4d(store: Rc<RefCell<FloatStorage>>, offset: usize,
                           size: [usize; 4], stride: [usize; 3])
     -> FloatTensor {
        unsafe {
            let tensor =
                THFloatTensor_newWithStorage4d(store.borrow_mut().storage(),
                                               offset, size[0], stride[0],
                                               size[1], stride[1], size[2],
                                               stride[2], size[3], 1);
            let dim = THFloatTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THFloatTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            FloatTensor{forget: false,
                        storage: Some(store),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_nd(store: Rc<RefCell<FloatStorage>>, offset: usize,
                           size: &[usize], stride: &[usize]) -> FloatTensor {
        if !(size.len() == stride.len() - 1 || size.len() == stride.len()) {
            {
                ::std::rt::begin_panic("Stride shall have either n - 1 elements or n elements where n = size.len()",
                                       &("tensor\\src\\lib.rs", 1293u32,
                                         1u32))
            }
        };
        let mut storage_len = 0;
        for i in 0..(stride.len() - 1) {
            let cur_len = stride[i] * size[i];
            if cur_len > storage_len { storage_len = cur_len; }
        }
        storage_len += size[size.len() - 1] - 1;
        let mut stride = stride.to_owned();
        if stride.len() == size.len() - 1 { stride.push(1); }
        unsafe {
            let tensor =
                THFloatTensor_newWithStorage1d(store.borrow_mut().storage(),
                                               offset, storage_len, 1);
            THFloatTensor_resizeNd(tensor, size.len() as i32,
                                   size.as_ptr() as *const i64,
                                   stride.as_ptr() as *const i64);
            let dim = THFloatTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THFloatTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            let stride = stride.to_vec();
            FloatTensor{forget: false,
                        storage: Some(store),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_1d(size: usize) -> FloatTensor {
        unsafe {
            let tensor = THFloatTensor_newWithSize1d(size as i64);
            let stride = THFloatTensor_stride(tensor, 0 as c_int) as usize;
            let storage =
                FloatStorage::from(THFloatTensor_storage(tensor)).forget();
            FloatTensor{forget: false,
                        storage: Some(Rc::new(RefCell::new(storage))),
                        tensor: tensor,
                        size: <[_]>::into_vec(box [size]),
                        stride: <[_]>::into_vec(box [stride]),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_2d(size: [usize; 2]) -> FloatTensor {
        unsafe {
            let tensor =
                THFloatTensor_newWithSize2d(size[0] as i64, size[1] as i64);
            let stride =
                [THFloatTensor_stride(tensor, 0 as c_int) as usize,
                 THFloatTensor_stride(tensor, 1 as c_int) as usize];
            let storage: FloatStorage = THFloatTensor_storage(tensor).into();
            let dim = THFloatTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THFloatTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            FloatTensor{forget: false,
                        storage:
                            Some(Rc::new(RefCell::new(storage.forget()))),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride.to_vec(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_3d(size: [usize; 3]) -> FloatTensor {
        unsafe {
            let tensor =
                THFloatTensor_newWithSize3d(size[0] as i64, size[1] as i64,
                                            size[2] as i64);
            let stride =
                [THFloatTensor_stride(tensor, 0 as c_int) as usize,
                 THFloatTensor_stride(tensor, 1 as c_int) as usize,
                 THFloatTensor_stride(tensor, 2 as c_int) as usize];
            let storage =
                FloatStorage::from(THFloatTensor_storage(tensor)).forget();
            let dim = THFloatTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THFloatTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            FloatTensor{forget: false,
                        storage: Some(Rc::new(RefCell::new(storage))),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride.to_vec(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_4d(size: [usize; 4]) -> FloatTensor {
        unsafe {
            let tensor =
                THFloatTensor_newWithSize4d(size[0] as i64, size[1] as i64,
                                            size[2] as i64, size[3] as i64);
            let stride =
                [THFloatTensor_stride(tensor, 0 as c_int) as usize,
                 THFloatTensor_stride(tensor, 1 as c_int) as usize,
                 THFloatTensor_stride(tensor, 2 as c_int) as usize,
                 THFloatTensor_stride(tensor, 3 as c_int) as usize];
            let storage =
                FloatStorage::from(THFloatTensor_storage(tensor)).forget();
            let dim = THFloatTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THFloatTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            FloatTensor{forget: false,
                        storage: Some(Rc::new(RefCell::new(storage))),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride.to_vec(),}
        }
    }
}
impl BasicManipulateOp<FloatStorage> for FloatTensor {
    type
    Datum
    =
    f32;
    fn desc(&self) -> String {
        unsafe { THFloatTensor_desc(self.tensor).to_string() }
    }
    fn dimensions(&self) -> usize {
        unsafe { THFloatTensor_nDimension(self.tensor) as usize }
    }
    fn get_0d(&self) -> f32 { unsafe { THFloatTensor_get0d(self.tensor) } }
    fn get_1d(&self, i: usize) -> f32 {
        unsafe { THFloatTensor_get1d(self.tensor, i as i64) }
    }
    fn get_2d(&self, i: [usize; 2]) -> f32 {
        unsafe { THFloatTensor_get2d(self.tensor, i[0] as i64, i[1] as i64) }
    }
    fn get_3d(&self, i: [usize; 3]) -> f32 {
        unsafe {
            THFloatTensor_get3d(self.tensor, i[0] as i64, i[1] as i64,
                                i[2] as i64)
        }
    }
    fn get_4d(&self, i: [usize; 4]) -> f32 {
        unsafe {
            THFloatTensor_get4d(self.tensor, i[0] as i64, i[1] as i64,
                                i[2] as i64, i[3] as i64)
        }
    }
    fn iter(&self) -> TensorIterator<f32> { self.into_iter() }
    fn iter_mut(&mut self) -> TensorIterMut<f32> { self.into_iter() }
    fn is_contiguous(&self) -> bool {
        unsafe { THFloatTensor_isContiguous(self.tensor) != 0 }
    }
    fn numel(&self) -> usize {
        unsafe { THFloatTensor_numel(self.tensor) as usize }
    }
    #[cfg(feature = "safe")]
    fn resize_0d(&mut self) {
        unsafe {
            self.size.clear();
            self.stride.clear();
            THFloatTensor_resize0d(self.tensor);
        }
    }
    #[cfg(feature = "safe")]
    fn resize_1d(&mut self, size: usize) {
        unsafe {
            self.size = <[_]>::into_vec(box [size]);
            self.stride = <[_]>::into_vec(box [1]);
            THFloatTensor_resize1d(self.tensor, size as i64);
            self.stride =
                <[_]>::into_vec(box
                                    [THFloatTensor_stride(self.tensor,
                                                          0 as c_int) as
                                         usize]);
        }
    }
    #[cfg(feature = "safe")]
    fn resize_2d(&mut self, size: [usize; 2]) {
        unsafe {
            self.size = size.to_vec();
            THFloatTensor_resize2d(self.tensor, size[0] as i64,
                                   size[1] as i64);
            self.stride =
                [THFloatTensor_stride(self.tensor, 0 as c_int) as usize,
                 THFloatTensor_stride(self.tensor, 1 as c_int) as
                     usize].to_vec();
        }
    }
    #[cfg(feature = "safe")]
    fn resize_3d(&mut self, size: [usize; 3]) {
        unsafe {
            self.size = size.to_vec();
            THFloatTensor_resize3d(self.tensor, size[0] as i64,
                                   size[1] as i64, size[2] as i64);
            self.stride =
                [THFloatTensor_stride(self.tensor, 0 as c_int) as usize,
                 THFloatTensor_stride(self.tensor, 1 as c_int) as usize,
                 THFloatTensor_stride(self.tensor, 2 as c_int) as
                     usize].to_vec();
        }
    }
    #[cfg(feature = "safe")]
    fn resize_nd(&mut self, size: &[usize], stride: &[usize]) {
        {
            match (&(size.len() - 1), &(stride.len())) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        {
                            ::std::rt::begin_panic_fmt(&::std::fmt::Arguments::new_v1(&["assertion failed: `(left == right)`\n  left: `",
                                                                                        "`,\n right: `",
                                                                                        "`: "],
                                                                                      &match (&left_val,
                                                                                              &right_val,
                                                                                              &::std::fmt::Arguments::new_v1(&["Stride must have exactly ",
                                                                                                                               " elements"],
                                                                                                                             &match (&(size.len()
                                                                                                                                           -
                                                                                                                                           1),)
                                                                                                                                  {
                                                                                                                                  (arg0,)
                                                                                                                                  =>
                                                                                                                                  [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                                                               ::std::fmt::Display::fmt)],
                                                                                                                              }))
                                                                                           {
                                                                                           (arg0,
                                                                                            arg1,
                                                                                            arg2)
                                                                                           =>
                                                                                           [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                        ::std::fmt::Debug::fmt),
                                                                                            ::std::fmt::ArgumentV1::new(arg1,
                                                                                                                        ::std::fmt::Debug::fmt),
                                                                                            ::std::fmt::ArgumentV1::new(arg2,
                                                                                                                        ::std::fmt::Display::fmt)],
                                                                                       }),
                                                       &("tensor\\src\\lib.rs",
                                                         1293u32, 1u32))
                        }
                    }
                }
            }
        };
        unsafe {
            self.size = size.to_owned();
            self.stride = stride.to_owned();
            self.stride.push(1);
            THFloatTensor_resizeNd(self.tensor, size.len() as c_int,
                                   size.as_ptr() as *const i64,
                                   self.stride.as_ptr() as *const i64);
        }
    }
    fn set_0d(&mut self, v: f32) {
        unsafe { THFloatTensor_set0d(self.tensor, v); }
    }
    fn set_1d(&mut self, i: usize, v: f32) {
        unsafe { THFloatTensor_set1d(self.tensor, i as i64, v); }
    }
    fn set_2d(&mut self, i: [usize; 2], v: f32) {
        unsafe {
            THFloatTensor_set2d(self.tensor, i[0] as i64, i[1] as i64, v);
        }
    }
    fn set_3d(&mut self, i: [usize; 3], v: f32) {
        unsafe {
            THFloatTensor_set3d(self.tensor, i[0] as i64, i[1] as i64,
                                i[2] as i64, v);
        }
    }
    fn set_4d(&mut self, i: [usize; 4], v: f32) {
        unsafe {
            THFloatTensor_set4d(self.tensor, i[0] as i64, i[1] as i64,
                                i[2] as i64, i[3] as i64, v);
        }
    }
    fn shape(&self) -> (&[usize], &[usize]) {
        (self.size.as_slice(), self.stride.as_slice())
    }
    fn size(&self, dim: usize) -> usize {
        if !(dim < self.size.len()) {
            {
                ::std::rt::begin_panic("assertion failed: dim < self.size.len()",
                                       &("tensor\\src\\lib.rs", 1293u32,
                                         1u32))
            }
        };
        unsafe { THFloatTensor_size(self.tensor, dim as c_int) as usize }
    }
    #[cfg(feature = "safe")]
    fn storage(&mut self) -> &mut Option<Rc<RefCell<FloatStorage>>> {
        &mut self.storage
    }
    fn storage_offset(&self) -> usize {
        unsafe { THFloatTensor_storageOffset(self.tensor) as usize }
    }
    fn stride(&self, dim: usize) -> usize {
        if !(dim < self.stride.len()) {
            {
                ::std::rt::begin_panic("assertion failed: dim < self.stride.len()",
                                       &("tensor\\src\\lib.rs", 1293u32,
                                         1u32))
            }
        };
        unsafe { THFloatTensor_stride(self.tensor, dim as c_int) as usize }
    }
}
impl UtilityOp<FloatStorage> for FloatTensor { }
impl ViewOp<FloatTensor> for FloatTensor {
    #[cfg(feature = "safe")]
    fn narrow(&self, bound: &[Range<usize>])
     -> Result<FloatTensor, NarrowError> {
        let (cur_shape, cur_stride) = self.shape();
        let mut new_size = Vec::with_capacity(bound.len());
        let mut offset = self.storage_offset();
        for (((dim, u_bound), cur_stride), new_bound) in
            cur_shape.iter().enumerate().zip(cur_stride.iter()).zip(bound.iter())
            {
            if *u_bound < new_bound.end {
                return Err(NarrowError{dim: dim,})
            } else {
                new_size.push(new_bound.end - new_bound.start);
                offset += new_bound.start * *cur_stride;
            }
        }
        let storage =
            Rc::clone(self.storage.as_ref().ok_or(NarrowError{dim: 0,})?);
        let tensor =
            FloatTensor::new_with_storage_nd(storage, offset, &new_size,
                                             &cur_stride);
        Ok(tensor)
    }
    #[cfg(feature = "safe")]
    fn narrow_on(&self, dim: usize, new_bound: Range<usize>)
     -> Result<FloatTensor, NarrowError> {
        let (cur_shape, _) = self.shape();
        if new_bound.end <= cur_shape[dim] {
        } else { return Err(NarrowError{dim: dim,}) }
        let tensor =
            self.new_narrow(dim, new_bound.start,
                            new_bound.end - new_bound.start);
        Ok(tensor)
    }
    #[cfg(feature = "safe")]
    fn squeeze(&self) -> FloatTensor {
        let mut new_ts = Self::new();
        new_ts.storage =
            match self.storage {
                Some(ref s) => Some(Rc::clone(s)),
                None => None,
            };
        unsafe {
            THFloatTensor_squeeze(new_ts.tensor, self.tensor);
            let dim = THFloatTensor_nDimension(new_ts.tensor);
            for i in 0..(dim - 1) {
                new_ts.size.push(THFloatTensor_size(new_ts.tensor, i as c_int)
                                     as usize);
                new_ts.stride.push(THFloatTensor_stride(new_ts.tensor,
                                                        i as c_int) as usize);
            }
            new_ts.size.push(THFloatTensor_size(new_ts.tensor,
                                                dim - 1 as c_int) as usize);
            new_ts.stride.push(THFloatTensor_stride(new_ts.tensor,
                                                    dim - 1 as c_int) as
                                   usize);
            new_ts
        }
    }
    #[cfg(feature = "safe")]
    fn view(&self, sizes: &[Option<usize>])
     -> Result<FloatTensor, ViewError> {
        let new_size = self.infer_size(sizes)?;
        let new_stride = self.compute_stride(&new_size)?;
        let offset = self.storage_offset();
        let mut storage =
            Rc::clone(self.storage.as_ref().ok_or(SizeInferError::ElementSizeMismatch)?);
        let ts =
            FloatTensor::new_with_storage_nd(storage, offset, &new_size,
                                             &new_stride);
        Ok(ts)
    }
}
impl Tensor for FloatTensor {
    type
    Datum
    =
    f32;
    type
    Storage
    =
    FloatStorage;
}
impl <'a> Clone for FloatTensor {
    #[cfg(feature = "safe")]
    fn clone(&self) -> FloatTensor {
        unsafe {
            let clone = THFloatTensor_newClone(self.tensor);
            let storage =
                Some(Rc::new(RefCell::new(FloatStorage::from(THFloatTensor_storage(clone)).forget())));
            let stride: Vec<usize> =
                (0..THFloatTensor_nDimension(clone)).map(|i|
                                                             {
                                                                 THFloatTensor_stride(clone,
                                                                                      i
                                                                                          as
                                                                                          i32)
                                                                     as usize
                                                             }).collect();
            FloatTensor{forget: false,
                        storage: storage,
                        tensor: clone,
                        size: self.size.to_owned(),
                        stride: stride,}
        }
    }
}
impl Drop for FloatTensor {
    fn drop(&mut self) {
        unsafe { if !self.forget { THFloatTensor_free(self.tensor); } }
    }
}
impl <'a> IntoIterator for &'a FloatTensor {
    type
    Item
    =
    f32;
    type
    IntoIter
    =
    TensorIterator<'a, f32>;
    fn into_iter(self) -> Self::IntoIter {
        let (size, stride) = self.shape();
        TensorIterator::new(self.data(), size, stride)
    }
}
impl <'a> IntoIterator for &'a mut FloatTensor {
    type
    Item
    =
    &'a mut f32;
    type
    IntoIter
    =
    TensorIterMut<'a, f32>;
    fn into_iter(self) -> Self::IntoIter {
        unsafe {
            let data = &mut *(self.data_mut() as *mut [f32]);
            let (size, stride) = self.shape();
            TensorIterMut::new(data, size, stride)
        }
    }
}
impl <'a> From<&'a [f32]> for FloatTensor {
    #[cfg(feature = "safe")]
    fn from(slice: &'a [f32]) -> FloatTensor {
        let mut tensor = FloatTensor::new_with_size_1d(slice.len());
        tensor.storage().borrow_mut().unwrap().iter_mut().enumerate().for_each(|(i,
                                                                                 v)|
                                                                                   *v
                                                                                       =
                                                                                       slice[i]);
        tensor
    }
}
use storage::THIntStorage;
#[repr(C)]
pub struct THIntTensor;
#[link(name = "caffe2")]
extern "C" {
    fn THIntTensor_new() -> *mut THIntTensor;
    fn THIntTensor_newClone(org: *const THIntTensor) -> *mut THIntTensor;
    fn THIntTensor_newContiguous(org: *const THIntTensor) -> *mut THIntTensor;
    fn THIntTensor_newNarrow(org: *const THIntTensor, dim: c_int, i: i64,
                             size: i64) -> *mut THIntTensor;
    fn THIntTensor_newSelect(org: *const THIntTensor, dim: c_int, idx: i64)
     -> *mut THIntTensor;
    fn THIntTensor_newTranspose(org: *const THIntTensor, dim_1: c_int,
                                dim_2: c_int) -> *mut THIntTensor;
    fn THIntTensor_newUnfold(org: *const THIntTensor, dim: c_int, size: i64,
                             step: i64) -> *mut THIntTensor;
    fn THIntTensor_newWithTensor(org: *const THIntTensor) -> *mut THIntTensor;
    fn THIntTensor_newWithStorage1d(store: *mut THIntStorage, offset: usize,
                                    size: usize, stride: usize)
     -> *mut THIntTensor;
    fn THIntTensor_newWithStorage2d(store: *mut THIntStorage, offset: usize,
                                    size_1: usize, stride_1: usize,
                                    size_2: usize, stride_2: usize)
     -> *mut THIntTensor;
    fn THIntTensor_newWithStorage3d(store: *mut THIntStorage, offset: usize,
                                    size_1: usize, stride_1: usize,
                                    size_2: usize, stride_2: usize,
                                    size_3: usize, stride_3: usize)
     -> *mut THIntTensor;
    fn THIntTensor_newWithStorage4d(store: *mut THIntStorage, offset: usize,
                                    size_1: usize, stride_1: usize,
                                    size_2: usize, stride_2: usize,
                                    size_3: usize, stride_3: usize,
                                    size_4: usize, stride_4: usize)
     -> *mut THIntTensor;
    fn THIntTensor_newWithSize1d(size: i64) -> *mut THIntTensor;
    fn THIntTensor_newWithSize2d(size_1: i64, size_2: i64)
     -> *mut THIntTensor;
    fn THIntTensor_newWithSize3d(size_1: i64, size_2: i64, size_3: i64)
     -> *mut THIntTensor;
    fn THIntTensor_newWithSize4d(size_1: i64, size_2: i64, size_3: i64,
                                 size_4: i64) -> *mut THIntTensor;
    fn THIntTensor_free(tensor: *mut THIntTensor);
    fn THIntTensor_data(tensor: *mut THIntTensor) -> *mut i32;
    fn THIntTensor_desc(tensor: *mut THIntTensor) -> THDescBuff;
    fn THIntTensor_nDimension(tensor: *const THIntTensor) -> c_int;
    fn THIntTensor_isContiguous(tensor: *const THIntTensor) -> c_int;
    fn THIntTensor_get0d(tensor: *const THIntTensor) -> i32;
    fn THIntTensor_get1d(tensor: *const THIntTensor, i: i64) -> i32;
    fn THIntTensor_get2d(tensor: *const THIntTensor, i: i64, j: i64) -> i32;
    fn THIntTensor_get3d(tensor: *const THIntTensor, i: i64, j: i64, k: i64)
     -> i32;
    fn THIntTensor_get4d(tensor: *const THIntTensor, i: i64, j: i64, k: i64,
                         l: i64) -> i32;
    fn THIntTensor_numel(tensor: *const THIntTensor) -> usize;
    fn THIntTensor_resize0d(tensor: *mut THIntTensor);
    fn THIntTensor_resize1d(tensor: *mut THIntTensor, size_1: i64);
    fn THIntTensor_resize2d(tensor: *mut THIntTensor, size_1: i64,
                            size_2: i64);
    fn THIntTensor_resize3d(tensor: *mut THIntTensor, size_1: i64,
                            size_2: i64, size_3: i64);
    fn THIntTensor_resize4d(tensor: *mut THIntTensor, size_1: i64,
                            size_2: i64, size_3: i64, size_4: i64);
    fn THIntTensor_resize5d(tensor: *mut THIntTensor, size_1: i64,
                            size_2: i64, size_3: i64, size_4: i64,
                            size_5: i64);
    fn THIntTensor_resizeNd(tensor: *mut THIntTensor, dim: c_int,
                            size: *const i64, stride: *const i64);
    fn THIntTensor_set0d(tensor: *const THIntTensor, v: i32);
    fn THIntTensor_set1d(tensor: *const THIntTensor, i: i64, v: i32);
    fn THIntTensor_set2d(tensor: *const THIntTensor, i: i64, j: i64, v: i32);
    fn THIntTensor_set3d(tensor: *const THIntTensor, i: i64, j: i64, k: i64,
                         v: i32);
    fn THIntTensor_set4d(tensor: *const THIntTensor, i: i64, j: i64, k: i64,
                         l: i64, v: i32);
    fn THIntTensor_size(tensor: *const THIntTensor, dim: c_int) -> i64;
    fn THIntTensor_setStorageNd(tensor: *const THIntTensor,
                                storage: THIntStorage, offset: i64,
                                dim: c_int, size: *const i64,
                                stride: *const i64);
    fn THIntTensor_storage(tensor: *mut THIntTensor) -> *mut THIntStorage;
    fn THIntTensor_storageOffset(tensor: *const THIntTensor) -> i64;
    fn THIntTensor_stride(tensor: *const THIntTensor, dim: c_int) -> i64;
    fn THIntTensor_squeeze(tensor: *mut THIntTensor, src: *const THIntTensor);
}
pub struct IntTensor {
    forget: bool,
    #[cfg(feature = "safe")]
    storage: Option<Rc<RefCell<IntStorage>>>,
    tensor: *mut THIntTensor,
    size: Vec<usize>,
    stride: Vec<usize>,
}
impl IntTensor {
    #[doc = r" Get short description of storage."]
    #[doc = r" This includes name of storage, size, and"]
    #[doc = r" sample data if it has more than 20 elements."]
    #[doc = r" If it has less than 20 elements, it'll display"]
    #[doc = r" every elements."]
    fn short_desc(&mut self) -> String {
        #[cfg(feature = "safe")]
        fn get_data(s: &IntTensor) -> &IntStorage {
            s.storage().as_ref().unwrap().borrow()
        }
        let size = self.size.as_slice();
        let stride = self.stride.as_slice();
        let data = get_data(self);
        let name = "IntTensor";
        if size.iter().fold(0, |cum, v| cum + v) > 20 {
            ::alloc::fmt::format(::std::fmt::Arguments::new_v1(&["", ":size=",
                                                                 ":stride=",
                                                                 ":first(10)=",
                                                                 ":last(10)="],
                                                               &match (&name,
                                                                       &size,
                                                                       &stride,
                                                                       &&(&*data)[0..10],
                                                                       &&(&*data)[(data.len()
                                                                                       -
                                                                                       10)..data.len()])
                                                                    {
                                                                    (arg0,
                                                                     arg1,
                                                                     arg2,
                                                                     arg3,
                                                                     arg4) =>
                                                                    [::std::fmt::ArgumentV1::new(arg0,
                                                                                                 ::std::fmt::Display::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg1,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg2,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg3,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg4,
                                                                                                 ::std::fmt::Debug::fmt)],
                                                                }))
        } else {
            ::alloc::fmt::format(::std::fmt::Arguments::new_v1(&["", ":size=",
                                                                 ":stride=",
                                                                 ":data="],
                                                               &match (&name,
                                                                       &size,
                                                                       &stride,
                                                                       &data.iter())
                                                                    {
                                                                    (arg0,
                                                                     arg1,
                                                                     arg2,
                                                                     arg3) =>
                                                                    [::std::fmt::ArgumentV1::new(arg0,
                                                                                                 ::std::fmt::Display::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg1,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg2,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg3,
                                                                                                 ::std::fmt::Debug::fmt)],
                                                                }))
        }
    }
    #[doc = r" Alias for short_desc"]
    #[inline(always)]
    fn to_string(&mut self) -> String { self.short_desc() }
}
impl CreateOp<IntStorage> for IntTensor {
    type
    Datum
    =
    i32;
    #[cfg(feature = "safe")]
    fn new() -> IntTensor {
        unsafe {
            IntTensor{forget: false,
                      storage: None,
                      tensor: THIntTensor_new(),
                      size: Vec::new(),
                      stride: Vec::new(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_contiguous(&self) -> IntTensor {
        unsafe {
            let cont = THIntTensor_newContiguous(self.tensor);
            let storage =
                Some(Rc::new(RefCell::new(IntStorage::from(THIntTensor_storage(cont)))));
            let stride: Vec<usize> =
                (0..THIntTensor_nDimension(cont)).map(|i|
                                                          {
                                                              THIntTensor_stride(cont,
                                                                                 i
                                                                                     as
                                                                                     i32)
                                                                  as usize
                                                          }).collect();
            IntTensor{forget: false,
                      storage: storage,
                      tensor: cont,
                      size: self.size.to_owned(),
                      stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_narrow(&self, dim: usize, i: usize, size: usize) -> IntTensor {
        unsafe {
            let tensor =
                THIntTensor_newNarrow(self.tensor, dim as c_int, i as i64,
                                      size as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THIntTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THIntTensor_stride(tensor, i as i32)
                                             as usize;
                                     let cur_size =
                                         THIntTensor_size(tensor, i as i32) as
                                             usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            IntTensor{forget: false,
                      storage: storage,
                      tensor: tensor,
                      size: size,
                      stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_select(&self, dim: usize, i: usize) -> IntTensor {
        unsafe {
            let tensor =
                THIntTensor_newSelect(self.tensor, dim as c_int, i as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THIntTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THIntTensor_stride(tensor, i as i32)
                                             as usize;
                                     let cur_size =
                                         THIntTensor_size(tensor, i as i32) as
                                             usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            IntTensor{forget: false,
                      storage: storage,
                      tensor: tensor,
                      size: size,
                      stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_transpose(&self, dim_1: usize, dim_2: usize) -> IntTensor {
        unsafe {
            let tensor =
                THIntTensor_newTranspose(self.tensor, dim_1 as c_int,
                                         dim_2 as c_int);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THIntTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THIntTensor_stride(tensor, i as i32)
                                             as usize;
                                     let cur_size =
                                         THIntTensor_size(tensor, i as i32) as
                                             usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            IntTensor{forget: false,
                      storage: storage,
                      tensor: tensor,
                      size: size,
                      stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_unfold(&self, dim: usize, size: usize, step: usize) -> IntTensor {
        unsafe {
            let tensor =
                THIntTensor_newUnfold(self.tensor, dim as c_int, size as i64,
                                      step as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THIntTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THIntTensor_stride(tensor, i as i32)
                                             as usize;
                                     let cur_size =
                                         THIntTensor_size(tensor, i as i32) as
                                             usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            IntTensor{forget: false,
                      storage: storage,
                      tensor: tensor,
                      size: size,
                      stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_1d(store: Rc<RefCell<IntStorage>>, offset: usize,
                           size: usize) -> IntTensor {
        unsafe {
            let tensor =
                THIntTensor_newWithStorage1d(store.borrow_mut().storage(),
                                             offset, size, 1);
            IntTensor{forget: false,
                      storage: Some(store),
                      tensor: tensor,
                      size: <[_]>::into_vec(box [size]),
                      stride: <[_]>::into_vec(box [1]),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_2d(store: Rc<RefCell<IntStorage>>, offset: usize,
                           size: [usize; 2], stride: usize) -> IntTensor {
        unsafe {
            let tensor =
                THIntTensor_newWithStorage2d(store.borrow_mut().storage(),
                                             offset, size[0], stride, size[1],
                                             1);
            let dim = THIntTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THIntTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            IntTensor{forget: false,
                      storage: Some(store),
                      tensor: tensor,
                      size: size.to_vec(),
                      stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_3d(store: Rc<RefCell<IntStorage>>, offset: usize,
                           size: [usize; 3], stride: [usize; 2])
     -> IntTensor {
        unsafe {
            let tensor =
                THIntTensor_newWithStorage3d(store.borrow_mut().storage(),
                                             offset, size[0], stride[0],
                                             size[1], stride[1], size[2], 1);
            let dim = THIntTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THIntTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            IntTensor{forget: false,
                      storage: Some(store),
                      tensor: tensor,
                      size: size.to_vec(),
                      stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_4d(store: Rc<RefCell<IntStorage>>, offset: usize,
                           size: [usize; 4], stride: [usize; 3])
     -> IntTensor {
        unsafe {
            let tensor =
                THIntTensor_newWithStorage4d(store.borrow_mut().storage(),
                                             offset, size[0], stride[0],
                                             size[1], stride[1], size[2],
                                             stride[2], size[3], 1);
            let dim = THIntTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THIntTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            IntTensor{forget: false,
                      storage: Some(store),
                      tensor: tensor,
                      size: size.to_vec(),
                      stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_nd(store: Rc<RefCell<IntStorage>>, offset: usize,
                           size: &[usize], stride: &[usize]) -> IntTensor {
        if !(size.len() == stride.len() - 1 || size.len() == stride.len()) {
            {
                ::std::rt::begin_panic("Stride shall have either n - 1 elements or n elements where n = size.len()",
                                       &("tensor\\src\\lib.rs", 1296u32,
                                         1u32))
            }
        };
        let mut storage_len = 0;
        for i in 0..(stride.len() - 1) {
            let cur_len = stride[i] * size[i];
            if cur_len > storage_len { storage_len = cur_len; }
        }
        storage_len += size[size.len() - 1] - 1;
        let mut stride = stride.to_owned();
        if stride.len() == size.len() - 1 { stride.push(1); }
        unsafe {
            let tensor =
                THIntTensor_newWithStorage1d(store.borrow_mut().storage(),
                                             offset, storage_len, 1);
            THIntTensor_resizeNd(tensor, size.len() as i32,
                                 size.as_ptr() as *const i64,
                                 stride.as_ptr() as *const i64);
            let dim = THIntTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THIntTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            let stride = stride.to_vec();
            IntTensor{forget: false,
                      storage: Some(store),
                      tensor: tensor,
                      size: size.to_vec(),
                      stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_1d(size: usize) -> IntTensor {
        unsafe {
            let tensor = THIntTensor_newWithSize1d(size as i64);
            let stride = THIntTensor_stride(tensor, 0 as c_int) as usize;
            let storage =
                IntStorage::from(THIntTensor_storage(tensor)).forget();
            IntTensor{forget: false,
                      storage: Some(Rc::new(RefCell::new(storage))),
                      tensor: tensor,
                      size: <[_]>::into_vec(box [size]),
                      stride: <[_]>::into_vec(box [stride]),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_2d(size: [usize; 2]) -> IntTensor {
        unsafe {
            let tensor =
                THIntTensor_newWithSize2d(size[0] as i64, size[1] as i64);
            let stride =
                [THIntTensor_stride(tensor, 0 as c_int) as usize,
                 THIntTensor_stride(tensor, 1 as c_int) as usize];
            let storage: IntStorage = THIntTensor_storage(tensor).into();
            let dim = THIntTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THIntTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            IntTensor{forget: false,
                      storage: Some(Rc::new(RefCell::new(storage.forget()))),
                      tensor: tensor,
                      size: size.to_vec(),
                      stride: stride.to_vec(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_3d(size: [usize; 3]) -> IntTensor {
        unsafe {
            let tensor =
                THIntTensor_newWithSize3d(size[0] as i64, size[1] as i64,
                                          size[2] as i64);
            let stride =
                [THIntTensor_stride(tensor, 0 as c_int) as usize,
                 THIntTensor_stride(tensor, 1 as c_int) as usize,
                 THIntTensor_stride(tensor, 2 as c_int) as usize];
            let storage =
                IntStorage::from(THIntTensor_storage(tensor)).forget();
            let dim = THIntTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THIntTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            IntTensor{forget: false,
                      storage: Some(Rc::new(RefCell::new(storage))),
                      tensor: tensor,
                      size: size.to_vec(),
                      stride: stride.to_vec(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_4d(size: [usize; 4]) -> IntTensor {
        unsafe {
            let tensor =
                THIntTensor_newWithSize4d(size[0] as i64, size[1] as i64,
                                          size[2] as i64, size[3] as i64);
            let stride =
                [THIntTensor_stride(tensor, 0 as c_int) as usize,
                 THIntTensor_stride(tensor, 1 as c_int) as usize,
                 THIntTensor_stride(tensor, 2 as c_int) as usize,
                 THIntTensor_stride(tensor, 3 as c_int) as usize];
            let storage =
                IntStorage::from(THIntTensor_storage(tensor)).forget();
            let dim = THIntTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THIntTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            IntTensor{forget: false,
                      storage: Some(Rc::new(RefCell::new(storage))),
                      tensor: tensor,
                      size: size.to_vec(),
                      stride: stride.to_vec(),}
        }
    }
}
impl BasicManipulateOp<IntStorage> for IntTensor {
    type
    Datum
    =
    i32;
    fn desc(&self) -> String {
        unsafe { THIntTensor_desc(self.tensor).to_string() }
    }
    fn dimensions(&self) -> usize {
        unsafe { THIntTensor_nDimension(self.tensor) as usize }
    }
    fn get_0d(&self) -> i32 { unsafe { THIntTensor_get0d(self.tensor) } }
    fn get_1d(&self, i: usize) -> i32 {
        unsafe { THIntTensor_get1d(self.tensor, i as i64) }
    }
    fn get_2d(&self, i: [usize; 2]) -> i32 {
        unsafe { THIntTensor_get2d(self.tensor, i[0] as i64, i[1] as i64) }
    }
    fn get_3d(&self, i: [usize; 3]) -> i32 {
        unsafe {
            THIntTensor_get3d(self.tensor, i[0] as i64, i[1] as i64,
                              i[2] as i64)
        }
    }
    fn get_4d(&self, i: [usize; 4]) -> i32 {
        unsafe {
            THIntTensor_get4d(self.tensor, i[0] as i64, i[1] as i64,
                              i[2] as i64, i[3] as i64)
        }
    }
    fn iter(&self) -> TensorIterator<i32> { self.into_iter() }
    fn iter_mut(&mut self) -> TensorIterMut<i32> { self.into_iter() }
    fn is_contiguous(&self) -> bool {
        unsafe { THIntTensor_isContiguous(self.tensor) != 0 }
    }
    fn numel(&self) -> usize {
        unsafe { THIntTensor_numel(self.tensor) as usize }
    }
    #[cfg(feature = "safe")]
    fn resize_0d(&mut self) {
        unsafe {
            self.size.clear();
            self.stride.clear();
            THIntTensor_resize0d(self.tensor);
        }
    }
    #[cfg(feature = "safe")]
    fn resize_1d(&mut self, size: usize) {
        unsafe {
            self.size = <[_]>::into_vec(box [size]);
            self.stride = <[_]>::into_vec(box [1]);
            THIntTensor_resize1d(self.tensor, size as i64);
            self.stride =
                <[_]>::into_vec(box
                                    [THIntTensor_stride(self.tensor,
                                                        0 as c_int) as
                                         usize]);
        }
    }
    #[cfg(feature = "safe")]
    fn resize_2d(&mut self, size: [usize; 2]) {
        unsafe {
            self.size = size.to_vec();
            THIntTensor_resize2d(self.tensor, size[0] as i64, size[1] as i64);
            self.stride =
                [THIntTensor_stride(self.tensor, 0 as c_int) as usize,
                 THIntTensor_stride(self.tensor, 1 as c_int) as
                     usize].to_vec();
        }
    }
    #[cfg(feature = "safe")]
    fn resize_3d(&mut self, size: [usize; 3]) {
        unsafe {
            self.size = size.to_vec();
            THIntTensor_resize3d(self.tensor, size[0] as i64, size[1] as i64,
                                 size[2] as i64);
            self.stride =
                [THIntTensor_stride(self.tensor, 0 as c_int) as usize,
                 THIntTensor_stride(self.tensor, 1 as c_int) as usize,
                 THIntTensor_stride(self.tensor, 2 as c_int) as
                     usize].to_vec();
        }
    }
    #[cfg(feature = "safe")]
    fn resize_nd(&mut self, size: &[usize], stride: &[usize]) {
        {
            match (&(size.len() - 1), &(stride.len())) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        {
                            ::std::rt::begin_panic_fmt(&::std::fmt::Arguments::new_v1(&["assertion failed: `(left == right)`\n  left: `",
                                                                                        "`,\n right: `",
                                                                                        "`: "],
                                                                                      &match (&left_val,
                                                                                              &right_val,
                                                                                              &::std::fmt::Arguments::new_v1(&["Stride must have exactly ",
                                                                                                                               " elements"],
                                                                                                                             &match (&(size.len()
                                                                                                                                           -
                                                                                                                                           1),)
                                                                                                                                  {
                                                                                                                                  (arg0,)
                                                                                                                                  =>
                                                                                                                                  [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                                                               ::std::fmt::Display::fmt)],
                                                                                                                              }))
                                                                                           {
                                                                                           (arg0,
                                                                                            arg1,
                                                                                            arg2)
                                                                                           =>
                                                                                           [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                        ::std::fmt::Debug::fmt),
                                                                                            ::std::fmt::ArgumentV1::new(arg1,
                                                                                                                        ::std::fmt::Debug::fmt),
                                                                                            ::std::fmt::ArgumentV1::new(arg2,
                                                                                                                        ::std::fmt::Display::fmt)],
                                                                                       }),
                                                       &("tensor\\src\\lib.rs",
                                                         1296u32, 1u32))
                        }
                    }
                }
            }
        };
        unsafe {
            self.size = size.to_owned();
            self.stride = stride.to_owned();
            self.stride.push(1);
            THIntTensor_resizeNd(self.tensor, size.len() as c_int,
                                 size.as_ptr() as *const i64,
                                 self.stride.as_ptr() as *const i64);
        }
    }
    fn set_0d(&mut self, v: i32) {
        unsafe { THIntTensor_set0d(self.tensor, v); }
    }
    fn set_1d(&mut self, i: usize, v: i32) {
        unsafe { THIntTensor_set1d(self.tensor, i as i64, v); }
    }
    fn set_2d(&mut self, i: [usize; 2], v: i32) {
        unsafe {
            THIntTensor_set2d(self.tensor, i[0] as i64, i[1] as i64, v);
        }
    }
    fn set_3d(&mut self, i: [usize; 3], v: i32) {
        unsafe {
            THIntTensor_set3d(self.tensor, i[0] as i64, i[1] as i64,
                              i[2] as i64, v);
        }
    }
    fn set_4d(&mut self, i: [usize; 4], v: i32) {
        unsafe {
            THIntTensor_set4d(self.tensor, i[0] as i64, i[1] as i64,
                              i[2] as i64, i[3] as i64, v);
        }
    }
    fn shape(&self) -> (&[usize], &[usize]) {
        (self.size.as_slice(), self.stride.as_slice())
    }
    fn size(&self, dim: usize) -> usize {
        if !(dim < self.size.len()) {
            {
                ::std::rt::begin_panic("assertion failed: dim < self.size.len()",
                                       &("tensor\\src\\lib.rs", 1296u32,
                                         1u32))
            }
        };
        unsafe { THIntTensor_size(self.tensor, dim as c_int) as usize }
    }
    #[cfg(feature = "safe")]
    fn storage(&mut self) -> &mut Option<Rc<RefCell<IntStorage>>> {
        &mut self.storage
    }
    fn storage_offset(&self) -> usize {
        unsafe { THIntTensor_storageOffset(self.tensor) as usize }
    }
    fn stride(&self, dim: usize) -> usize {
        if !(dim < self.stride.len()) {
            {
                ::std::rt::begin_panic("assertion failed: dim < self.stride.len()",
                                       &("tensor\\src\\lib.rs", 1296u32,
                                         1u32))
            }
        };
        unsafe { THIntTensor_stride(self.tensor, dim as c_int) as usize }
    }
}
impl UtilityOp<IntStorage> for IntTensor { }
impl ViewOp<IntTensor> for IntTensor {
    #[cfg(feature = "safe")]
    fn narrow(&self, bound: &[Range<usize>])
     -> Result<IntTensor, NarrowError> {
        let (cur_shape, cur_stride) = self.shape();
        let mut new_size = Vec::with_capacity(bound.len());
        let mut offset = self.storage_offset();
        for (((dim, u_bound), cur_stride), new_bound) in
            cur_shape.iter().enumerate().zip(cur_stride.iter()).zip(bound.iter())
            {
            if *u_bound < new_bound.end {
                return Err(NarrowError{dim: dim,})
            } else {
                new_size.push(new_bound.end - new_bound.start);
                offset += new_bound.start * *cur_stride;
            }
        }
        let storage =
            Rc::clone(self.storage.as_ref().ok_or(NarrowError{dim: 0,})?);
        let tensor =
            IntTensor::new_with_storage_nd(storage, offset, &new_size,
                                           &cur_stride);
        Ok(tensor)
    }
    #[cfg(feature = "safe")]
    fn narrow_on(&self, dim: usize, new_bound: Range<usize>)
     -> Result<IntTensor, NarrowError> {
        let (cur_shape, _) = self.shape();
        if new_bound.end <= cur_shape[dim] {
        } else { return Err(NarrowError{dim: dim,}) }
        let tensor =
            self.new_narrow(dim, new_bound.start,
                            new_bound.end - new_bound.start);
        Ok(tensor)
    }
    #[cfg(feature = "safe")]
    fn squeeze(&self) -> IntTensor {
        let mut new_ts = Self::new();
        new_ts.storage =
            match self.storage {
                Some(ref s) => Some(Rc::clone(s)),
                None => None,
            };
        unsafe {
            THIntTensor_squeeze(new_ts.tensor, self.tensor);
            let dim = THIntTensor_nDimension(new_ts.tensor);
            for i in 0..(dim - 1) {
                new_ts.size.push(THIntTensor_size(new_ts.tensor, i as c_int)
                                     as usize);
                new_ts.stride.push(THIntTensor_stride(new_ts.tensor,
                                                      i as c_int) as usize);
            }
            new_ts.size.push(THIntTensor_size(new_ts.tensor, dim - 1 as c_int)
                                 as usize);
            new_ts.stride.push(THIntTensor_stride(new_ts.tensor,
                                                  dim - 1 as c_int) as usize);
            new_ts
        }
    }
    #[cfg(feature = "safe")]
    fn view(&self, sizes: &[Option<usize>]) -> Result<IntTensor, ViewError> {
        let new_size = self.infer_size(sizes)?;
        let new_stride = self.compute_stride(&new_size)?;
        let offset = self.storage_offset();
        let mut storage =
            Rc::clone(self.storage.as_ref().ok_or(SizeInferError::ElementSizeMismatch)?);
        let ts =
            IntTensor::new_with_storage_nd(storage, offset, &new_size,
                                           &new_stride);
        Ok(ts)
    }
}
impl Tensor for IntTensor {
    type
    Datum
    =
    i32;
    type
    Storage
    =
    IntStorage;
}
impl <'a> Clone for IntTensor {
    #[cfg(feature = "safe")]
    fn clone(&self) -> IntTensor {
        unsafe {
            let clone = THIntTensor_newClone(self.tensor);
            let storage =
                Some(Rc::new(RefCell::new(IntStorage::from(THIntTensor_storage(clone)).forget())));
            let stride: Vec<usize> =
                (0..THIntTensor_nDimension(clone)).map(|i|
                                                           {
                                                               THIntTensor_stride(clone,
                                                                                  i
                                                                                      as
                                                                                      i32)
                                                                   as usize
                                                           }).collect();
            IntTensor{forget: false,
                      storage: storage,
                      tensor: clone,
                      size: self.size.to_owned(),
                      stride: stride,}
        }
    }
}
impl Drop for IntTensor {
    fn drop(&mut self) {
        unsafe { if !self.forget { THIntTensor_free(self.tensor); } }
    }
}
impl <'a> IntoIterator for &'a IntTensor {
    type
    Item
    =
    i32;
    type
    IntoIter
    =
    TensorIterator<'a, i32>;
    fn into_iter(self) -> Self::IntoIter {
        let (size, stride) = self.shape();
        TensorIterator::new(self.data(), size, stride)
    }
}
impl <'a> IntoIterator for &'a mut IntTensor {
    type
    Item
    =
    &'a mut i32;
    type
    IntoIter
    =
    TensorIterMut<'a, i32>;
    fn into_iter(self) -> Self::IntoIter {
        unsafe {
            let data = &mut *(self.data_mut() as *mut [i32]);
            let (size, stride) = self.shape();
            TensorIterMut::new(data, size, stride)
        }
    }
}
impl <'a> From<&'a [i32]> for IntTensor {
    #[cfg(feature = "safe")]
    fn from(slice: &'a [i32]) -> IntTensor {
        let mut tensor = IntTensor::new_with_size_1d(slice.len());
        tensor.storage().borrow_mut().unwrap().iter_mut().enumerate().for_each(|(i,
                                                                                 v)|
                                                                                   *v
                                                                                       =
                                                                                       slice[i]);
        tensor
    }
}
use storage::THLongStorage;
#[repr(C)]
pub struct THLongTensor;
#[link(name = "caffe2")]
extern "C" {
    fn THLongTensor_new() -> *mut THLongTensor;
    fn THLongTensor_newClone(org: *const THLongTensor) -> *mut THLongTensor;
    fn THLongTensor_newContiguous(org: *const THLongTensor)
     -> *mut THLongTensor;
    fn THLongTensor_newNarrow(org: *const THLongTensor, dim: c_int, i: i64,
                              size: i64) -> *mut THLongTensor;
    fn THLongTensor_newSelect(org: *const THLongTensor, dim: c_int, idx: i64)
     -> *mut THLongTensor;
    fn THLongTensor_newTranspose(org: *const THLongTensor, dim_1: c_int,
                                 dim_2: c_int) -> *mut THLongTensor;
    fn THLongTensor_newUnfold(org: *const THLongTensor, dim: c_int, size: i64,
                              step: i64) -> *mut THLongTensor;
    fn THLongTensor_newWithTensor(org: *const THLongTensor)
     -> *mut THLongTensor;
    fn THLongTensor_newWithStorage1d(store: *mut THLongStorage, offset: usize,
                                     size: usize, stride: usize)
     -> *mut THLongTensor;
    fn THLongTensor_newWithStorage2d(store: *mut THLongStorage, offset: usize,
                                     size_1: usize, stride_1: usize,
                                     size_2: usize, stride_2: usize)
     -> *mut THLongTensor;
    fn THLongTensor_newWithStorage3d(store: *mut THLongStorage, offset: usize,
                                     size_1: usize, stride_1: usize,
                                     size_2: usize, stride_2: usize,
                                     size_3: usize, stride_3: usize)
     -> *mut THLongTensor;
    fn THLongTensor_newWithStorage4d(store: *mut THLongStorage, offset: usize,
                                     size_1: usize, stride_1: usize,
                                     size_2: usize, stride_2: usize,
                                     size_3: usize, stride_3: usize,
                                     size_4: usize, stride_4: usize)
     -> *mut THLongTensor;
    fn THLongTensor_newWithSize1d(size: i64) -> *mut THLongTensor;
    fn THLongTensor_newWithSize2d(size_1: i64, size_2: i64)
     -> *mut THLongTensor;
    fn THLongTensor_newWithSize3d(size_1: i64, size_2: i64, size_3: i64)
     -> *mut THLongTensor;
    fn THLongTensor_newWithSize4d(size_1: i64, size_2: i64, size_3: i64,
                                  size_4: i64) -> *mut THLongTensor;
    fn THLongTensor_free(tensor: *mut THLongTensor);
    fn THLongTensor_data(tensor: *mut THLongTensor) -> *mut i64;
    fn THLongTensor_desc(tensor: *mut THLongTensor) -> THDescBuff;
    fn THLongTensor_nDimension(tensor: *const THLongTensor) -> c_int;
    fn THLongTensor_isContiguous(tensor: *const THLongTensor) -> c_int;
    fn THLongTensor_get0d(tensor: *const THLongTensor) -> i64;
    fn THLongTensor_get1d(tensor: *const THLongTensor, i: i64) -> i64;
    fn THLongTensor_get2d(tensor: *const THLongTensor, i: i64, j: i64) -> i64;
    fn THLongTensor_get3d(tensor: *const THLongTensor, i: i64, j: i64, k: i64)
     -> i64;
    fn THLongTensor_get4d(tensor: *const THLongTensor, i: i64, j: i64, k: i64,
                          l: i64) -> i64;
    fn THLongTensor_numel(tensor: *const THLongTensor) -> usize;
    fn THLongTensor_resize0d(tensor: *mut THLongTensor);
    fn THLongTensor_resize1d(tensor: *mut THLongTensor, size_1: i64);
    fn THLongTensor_resize2d(tensor: *mut THLongTensor, size_1: i64,
                             size_2: i64);
    fn THLongTensor_resize3d(tensor: *mut THLongTensor, size_1: i64,
                             size_2: i64, size_3: i64);
    fn THLongTensor_resize4d(tensor: *mut THLongTensor, size_1: i64,
                             size_2: i64, size_3: i64, size_4: i64);
    fn THLongTensor_resize5d(tensor: *mut THLongTensor, size_1: i64,
                             size_2: i64, size_3: i64, size_4: i64,
                             size_5: i64);
    fn THLongTensor_resizeNd(tensor: *mut THLongTensor, dim: c_int,
                             size: *const i64, stride: *const i64);
    fn THLongTensor_set0d(tensor: *const THLongTensor, v: i64);
    fn THLongTensor_set1d(tensor: *const THLongTensor, i: i64, v: i64);
    fn THLongTensor_set2d(tensor: *const THLongTensor, i: i64, j: i64,
                          v: i64);
    fn THLongTensor_set3d(tensor: *const THLongTensor, i: i64, j: i64, k: i64,
                          v: i64);
    fn THLongTensor_set4d(tensor: *const THLongTensor, i: i64, j: i64, k: i64,
                          l: i64, v: i64);
    fn THLongTensor_size(tensor: *const THLongTensor, dim: c_int) -> i64;
    fn THLongTensor_setStorageNd(tensor: *const THLongTensor,
                                 storage: THLongStorage, offset: i64,
                                 dim: c_int, size: *const i64,
                                 stride: *const i64);
    fn THLongTensor_storage(tensor: *mut THLongTensor) -> *mut THLongStorage;
    fn THLongTensor_storageOffset(tensor: *const THLongTensor) -> i64;
    fn THLongTensor_stride(tensor: *const THLongTensor, dim: c_int) -> i64;
    fn THLongTensor_squeeze(tensor: *mut THLongTensor,
                            src: *const THLongTensor);
}
pub struct LongTensor {
    forget: bool,
    #[cfg(feature = "safe")]
    storage: Option<Rc<RefCell<LongStorage>>>,
    tensor: *mut THLongTensor,
    size: Vec<usize>,
    stride: Vec<usize>,
}
impl LongTensor {
    #[doc = r" Get short description of storage."]
    #[doc = r" This includes name of storage, size, and"]
    #[doc = r" sample data if it has more than 20 elements."]
    #[doc = r" If it has less than 20 elements, it'll display"]
    #[doc = r" every elements."]
    fn short_desc(&mut self) -> String {
        #[cfg(feature = "safe")]
        fn get_data(s: &LongTensor) -> &LongStorage {
            s.storage().as_ref().unwrap().borrow()
        }
        let size = self.size.as_slice();
        let stride = self.stride.as_slice();
        let data = get_data(self);
        let name = "LongTensor";
        if size.iter().fold(0, |cum, v| cum + v) > 20 {
            ::alloc::fmt::format(::std::fmt::Arguments::new_v1(&["", ":size=",
                                                                 ":stride=",
                                                                 ":first(10)=",
                                                                 ":last(10)="],
                                                               &match (&name,
                                                                       &size,
                                                                       &stride,
                                                                       &&(&*data)[0..10],
                                                                       &&(&*data)[(data.len()
                                                                                       -
                                                                                       10)..data.len()])
                                                                    {
                                                                    (arg0,
                                                                     arg1,
                                                                     arg2,
                                                                     arg3,
                                                                     arg4) =>
                                                                    [::std::fmt::ArgumentV1::new(arg0,
                                                                                                 ::std::fmt::Display::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg1,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg2,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg3,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg4,
                                                                                                 ::std::fmt::Debug::fmt)],
                                                                }))
        } else {
            ::alloc::fmt::format(::std::fmt::Arguments::new_v1(&["", ":size=",
                                                                 ":stride=",
                                                                 ":data="],
                                                               &match (&name,
                                                                       &size,
                                                                       &stride,
                                                                       &data.iter())
                                                                    {
                                                                    (arg0,
                                                                     arg1,
                                                                     arg2,
                                                                     arg3) =>
                                                                    [::std::fmt::ArgumentV1::new(arg0,
                                                                                                 ::std::fmt::Display::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg1,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg2,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg3,
                                                                                                 ::std::fmt::Debug::fmt)],
                                                                }))
        }
    }
    #[doc = r" Alias for short_desc"]
    #[inline(always)]
    fn to_string(&mut self) -> String { self.short_desc() }
}
impl CreateOp<LongStorage> for LongTensor {
    type
    Datum
    =
    i64;
    #[cfg(feature = "safe")]
    fn new() -> LongTensor {
        unsafe {
            LongTensor{forget: false,
                       storage: None,
                       tensor: THLongTensor_new(),
                       size: Vec::new(),
                       stride: Vec::new(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_contiguous(&self) -> LongTensor {
        unsafe {
            let cont = THLongTensor_newContiguous(self.tensor);
            let storage =
                Some(Rc::new(RefCell::new(LongStorage::from(THLongTensor_storage(cont)))));
            let stride: Vec<usize> =
                (0..THLongTensor_nDimension(cont)).map(|i|
                                                           {
                                                               THLongTensor_stride(cont,
                                                                                   i
                                                                                       as
                                                                                       i32)
                                                                   as usize
                                                           }).collect();
            LongTensor{forget: false,
                       storage: storage,
                       tensor: cont,
                       size: self.size.to_owned(),
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_narrow(&self, dim: usize, i: usize, size: usize) -> LongTensor {
        unsafe {
            let tensor =
                THLongTensor_newNarrow(self.tensor, dim as c_int, i as i64,
                                       size as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THLongTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THLongTensor_stride(tensor, i as i32)
                                             as usize;
                                     let cur_size =
                                         THLongTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            LongTensor{forget: false,
                       storage: storage,
                       tensor: tensor,
                       size: size,
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_select(&self, dim: usize, i: usize) -> LongTensor {
        unsafe {
            let tensor =
                THLongTensor_newSelect(self.tensor, dim as c_int, i as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THLongTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THLongTensor_stride(tensor, i as i32)
                                             as usize;
                                     let cur_size =
                                         THLongTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            LongTensor{forget: false,
                       storage: storage,
                       tensor: tensor,
                       size: size,
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_transpose(&self, dim_1: usize, dim_2: usize) -> LongTensor {
        unsafe {
            let tensor =
                THLongTensor_newTranspose(self.tensor, dim_1 as c_int,
                                          dim_2 as c_int);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THLongTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THLongTensor_stride(tensor, i as i32)
                                             as usize;
                                     let cur_size =
                                         THLongTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            LongTensor{forget: false,
                       storage: storage,
                       tensor: tensor,
                       size: size,
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_unfold(&self, dim: usize, size: usize, step: usize) -> LongTensor {
        unsafe {
            let tensor =
                THLongTensor_newUnfold(self.tensor, dim as c_int, size as i64,
                                       step as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THLongTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THLongTensor_stride(tensor, i as i32)
                                             as usize;
                                     let cur_size =
                                         THLongTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            LongTensor{forget: false,
                       storage: storage,
                       tensor: tensor,
                       size: size,
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_1d(store: Rc<RefCell<LongStorage>>, offset: usize,
                           size: usize) -> LongTensor {
        unsafe {
            let tensor =
                THLongTensor_newWithStorage1d(store.borrow_mut().storage(),
                                              offset, size, 1);
            LongTensor{forget: false,
                       storage: Some(store),
                       tensor: tensor,
                       size: <[_]>::into_vec(box [size]),
                       stride: <[_]>::into_vec(box [1]),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_2d(store: Rc<RefCell<LongStorage>>, offset: usize,
                           size: [usize; 2], stride: usize) -> LongTensor {
        unsafe {
            let tensor =
                THLongTensor_newWithStorage2d(store.borrow_mut().storage(),
                                              offset, size[0], stride,
                                              size[1], 1);
            let dim = THLongTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THLongTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            LongTensor{forget: false,
                       storage: Some(store),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_3d(store: Rc<RefCell<LongStorage>>, offset: usize,
                           size: [usize; 3], stride: [usize; 2])
     -> LongTensor {
        unsafe {
            let tensor =
                THLongTensor_newWithStorage3d(store.borrow_mut().storage(),
                                              offset, size[0], stride[0],
                                              size[1], stride[1], size[2], 1);
            let dim = THLongTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THLongTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            LongTensor{forget: false,
                       storage: Some(store),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_4d(store: Rc<RefCell<LongStorage>>, offset: usize,
                           size: [usize; 4], stride: [usize; 3])
     -> LongTensor {
        unsafe {
            let tensor =
                THLongTensor_newWithStorage4d(store.borrow_mut().storage(),
                                              offset, size[0], stride[0],
                                              size[1], stride[1], size[2],
                                              stride[2], size[3], 1);
            let dim = THLongTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THLongTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            LongTensor{forget: false,
                       storage: Some(store),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_nd(store: Rc<RefCell<LongStorage>>, offset: usize,
                           size: &[usize], stride: &[usize]) -> LongTensor {
        if !(size.len() == stride.len() - 1 || size.len() == stride.len()) {
            {
                ::std::rt::begin_panic("Stride shall have either n - 1 elements or n elements where n = size.len()",
                                       &("tensor\\src\\lib.rs", 1298u32,
                                         1u32))
            }
        };
        let mut storage_len = 0;
        for i in 0..(stride.len() - 1) {
            let cur_len = stride[i] * size[i];
            if cur_len > storage_len { storage_len = cur_len; }
        }
        storage_len += size[size.len() - 1] - 1;
        let mut stride = stride.to_owned();
        if stride.len() == size.len() - 1 { stride.push(1); }
        unsafe {
            let tensor =
                THLongTensor_newWithStorage1d(store.borrow_mut().storage(),
                                              offset, storage_len, 1);
            THLongTensor_resizeNd(tensor, size.len() as i32,
                                  size.as_ptr() as *const i64,
                                  stride.as_ptr() as *const i64);
            let dim = THLongTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THLongTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            let stride = stride.to_vec();
            LongTensor{forget: false,
                       storage: Some(store),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_1d(size: usize) -> LongTensor {
        unsafe {
            let tensor = THLongTensor_newWithSize1d(size as i64);
            let stride = THLongTensor_stride(tensor, 0 as c_int) as usize;
            let storage =
                LongStorage::from(THLongTensor_storage(tensor)).forget();
            LongTensor{forget: false,
                       storage: Some(Rc::new(RefCell::new(storage))),
                       tensor: tensor,
                       size: <[_]>::into_vec(box [size]),
                       stride: <[_]>::into_vec(box [stride]),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_2d(size: [usize; 2]) -> LongTensor {
        unsafe {
            let tensor =
                THLongTensor_newWithSize2d(size[0] as i64, size[1] as i64);
            let stride =
                [THLongTensor_stride(tensor, 0 as c_int) as usize,
                 THLongTensor_stride(tensor, 1 as c_int) as usize];
            let storage: LongStorage = THLongTensor_storage(tensor).into();
            let dim = THLongTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THLongTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            LongTensor{forget: false,
                       storage: Some(Rc::new(RefCell::new(storage.forget()))),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride.to_vec(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_3d(size: [usize; 3]) -> LongTensor {
        unsafe {
            let tensor =
                THLongTensor_newWithSize3d(size[0] as i64, size[1] as i64,
                                           size[2] as i64);
            let stride =
                [THLongTensor_stride(tensor, 0 as c_int) as usize,
                 THLongTensor_stride(tensor, 1 as c_int) as usize,
                 THLongTensor_stride(tensor, 2 as c_int) as usize];
            let storage =
                LongStorage::from(THLongTensor_storage(tensor)).forget();
            let dim = THLongTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THLongTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            LongTensor{forget: false,
                       storage: Some(Rc::new(RefCell::new(storage))),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride.to_vec(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_4d(size: [usize; 4]) -> LongTensor {
        unsafe {
            let tensor =
                THLongTensor_newWithSize4d(size[0] as i64, size[1] as i64,
                                           size[2] as i64, size[3] as i64);
            let stride =
                [THLongTensor_stride(tensor, 0 as c_int) as usize,
                 THLongTensor_stride(tensor, 1 as c_int) as usize,
                 THLongTensor_stride(tensor, 2 as c_int) as usize,
                 THLongTensor_stride(tensor, 3 as c_int) as usize];
            let storage =
                LongStorage::from(THLongTensor_storage(tensor)).forget();
            let dim = THLongTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THLongTensor_stride(tensor, i as i32)
                                             as usize;
                                     cur_stride
                                 }).collect();
            LongTensor{forget: false,
                       storage: Some(Rc::new(RefCell::new(storage))),
                       tensor: tensor,
                       size: size.to_vec(),
                       stride: stride.to_vec(),}
        }
    }
}
impl BasicManipulateOp<LongStorage> for LongTensor {
    type
    Datum
    =
    i64;
    fn desc(&self) -> String {
        unsafe { THLongTensor_desc(self.tensor).to_string() }
    }
    fn dimensions(&self) -> usize {
        unsafe { THLongTensor_nDimension(self.tensor) as usize }
    }
    fn get_0d(&self) -> i64 { unsafe { THLongTensor_get0d(self.tensor) } }
    fn get_1d(&self, i: usize) -> i64 {
        unsafe { THLongTensor_get1d(self.tensor, i as i64) }
    }
    fn get_2d(&self, i: [usize; 2]) -> i64 {
        unsafe { THLongTensor_get2d(self.tensor, i[0] as i64, i[1] as i64) }
    }
    fn get_3d(&self, i: [usize; 3]) -> i64 {
        unsafe {
            THLongTensor_get3d(self.tensor, i[0] as i64, i[1] as i64,
                               i[2] as i64)
        }
    }
    fn get_4d(&self, i: [usize; 4]) -> i64 {
        unsafe {
            THLongTensor_get4d(self.tensor, i[0] as i64, i[1] as i64,
                               i[2] as i64, i[3] as i64)
        }
    }
    fn iter(&self) -> TensorIterator<i64> { self.into_iter() }
    fn iter_mut(&mut self) -> TensorIterMut<i64> { self.into_iter() }
    fn is_contiguous(&self) -> bool {
        unsafe { THLongTensor_isContiguous(self.tensor) != 0 }
    }
    fn numel(&self) -> usize {
        unsafe { THLongTensor_numel(self.tensor) as usize }
    }
    #[cfg(feature = "safe")]
    fn resize_0d(&mut self) {
        unsafe {
            self.size.clear();
            self.stride.clear();
            THLongTensor_resize0d(self.tensor);
        }
    }
    #[cfg(feature = "safe")]
    fn resize_1d(&mut self, size: usize) {
        unsafe {
            self.size = <[_]>::into_vec(box [size]);
            self.stride = <[_]>::into_vec(box [1]);
            THLongTensor_resize1d(self.tensor, size as i64);
            self.stride =
                <[_]>::into_vec(box
                                    [THLongTensor_stride(self.tensor,
                                                         0 as c_int) as
                                         usize]);
        }
    }
    #[cfg(feature = "safe")]
    fn resize_2d(&mut self, size: [usize; 2]) {
        unsafe {
            self.size = size.to_vec();
            THLongTensor_resize2d(self.tensor, size[0] as i64,
                                  size[1] as i64);
            self.stride =
                [THLongTensor_stride(self.tensor, 0 as c_int) as usize,
                 THLongTensor_stride(self.tensor, 1 as c_int) as
                     usize].to_vec();
        }
    }
    #[cfg(feature = "safe")]
    fn resize_3d(&mut self, size: [usize; 3]) {
        unsafe {
            self.size = size.to_vec();
            THLongTensor_resize3d(self.tensor, size[0] as i64, size[1] as i64,
                                  size[2] as i64);
            self.stride =
                [THLongTensor_stride(self.tensor, 0 as c_int) as usize,
                 THLongTensor_stride(self.tensor, 1 as c_int) as usize,
                 THLongTensor_stride(self.tensor, 2 as c_int) as
                     usize].to_vec();
        }
    }
    #[cfg(feature = "safe")]
    fn resize_nd(&mut self, size: &[usize], stride: &[usize]) {
        {
            match (&(size.len() - 1), &(stride.len())) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        {
                            ::std::rt::begin_panic_fmt(&::std::fmt::Arguments::new_v1(&["assertion failed: `(left == right)`\n  left: `",
                                                                                        "`,\n right: `",
                                                                                        "`: "],
                                                                                      &match (&left_val,
                                                                                              &right_val,
                                                                                              &::std::fmt::Arguments::new_v1(&["Stride must have exactly ",
                                                                                                                               " elements"],
                                                                                                                             &match (&(size.len()
                                                                                                                                           -
                                                                                                                                           1),)
                                                                                                                                  {
                                                                                                                                  (arg0,)
                                                                                                                                  =>
                                                                                                                                  [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                                                               ::std::fmt::Display::fmt)],
                                                                                                                              }))
                                                                                           {
                                                                                           (arg0,
                                                                                            arg1,
                                                                                            arg2)
                                                                                           =>
                                                                                           [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                        ::std::fmt::Debug::fmt),
                                                                                            ::std::fmt::ArgumentV1::new(arg1,
                                                                                                                        ::std::fmt::Debug::fmt),
                                                                                            ::std::fmt::ArgumentV1::new(arg2,
                                                                                                                        ::std::fmt::Display::fmt)],
                                                                                       }),
                                                       &("tensor\\src\\lib.rs",
                                                         1298u32, 1u32))
                        }
                    }
                }
            }
        };
        unsafe {
            self.size = size.to_owned();
            self.stride = stride.to_owned();
            self.stride.push(1);
            THLongTensor_resizeNd(self.tensor, size.len() as c_int,
                                  size.as_ptr() as *const i64,
                                  self.stride.as_ptr() as *const i64);
        }
    }
    fn set_0d(&mut self, v: i64) {
        unsafe { THLongTensor_set0d(self.tensor, v); }
    }
    fn set_1d(&mut self, i: usize, v: i64) {
        unsafe { THLongTensor_set1d(self.tensor, i as i64, v); }
    }
    fn set_2d(&mut self, i: [usize; 2], v: i64) {
        unsafe {
            THLongTensor_set2d(self.tensor, i[0] as i64, i[1] as i64, v);
        }
    }
    fn set_3d(&mut self, i: [usize; 3], v: i64) {
        unsafe {
            THLongTensor_set3d(self.tensor, i[0] as i64, i[1] as i64,
                               i[2] as i64, v);
        }
    }
    fn set_4d(&mut self, i: [usize; 4], v: i64) {
        unsafe {
            THLongTensor_set4d(self.tensor, i[0] as i64, i[1] as i64,
                               i[2] as i64, i[3] as i64, v);
        }
    }
    fn shape(&self) -> (&[usize], &[usize]) {
        (self.size.as_slice(), self.stride.as_slice())
    }
    fn size(&self, dim: usize) -> usize {
        if !(dim < self.size.len()) {
            {
                ::std::rt::begin_panic("assertion failed: dim < self.size.len()",
                                       &("tensor\\src\\lib.rs", 1298u32,
                                         1u32))
            }
        };
        unsafe { THLongTensor_size(self.tensor, dim as c_int) as usize }
    }
    #[cfg(feature = "safe")]
    fn storage(&mut self) -> &mut Option<Rc<RefCell<LongStorage>>> {
        &mut self.storage
    }
    fn storage_offset(&self) -> usize {
        unsafe { THLongTensor_storageOffset(self.tensor) as usize }
    }
    fn stride(&self, dim: usize) -> usize {
        if !(dim < self.stride.len()) {
            {
                ::std::rt::begin_panic("assertion failed: dim < self.stride.len()",
                                       &("tensor\\src\\lib.rs", 1298u32,
                                         1u32))
            }
        };
        unsafe { THLongTensor_stride(self.tensor, dim as c_int) as usize }
    }
}
impl UtilityOp<LongStorage> for LongTensor { }
impl ViewOp<LongTensor> for LongTensor {
    #[cfg(feature = "safe")]
    fn narrow(&self, bound: &[Range<usize>])
     -> Result<LongTensor, NarrowError> {
        let (cur_shape, cur_stride) = self.shape();
        let mut new_size = Vec::with_capacity(bound.len());
        let mut offset = self.storage_offset();
        for (((dim, u_bound), cur_stride), new_bound) in
            cur_shape.iter().enumerate().zip(cur_stride.iter()).zip(bound.iter())
            {
            if *u_bound < new_bound.end {
                return Err(NarrowError{dim: dim,})
            } else {
                new_size.push(new_bound.end - new_bound.start);
                offset += new_bound.start * *cur_stride;
            }
        }
        let storage =
            Rc::clone(self.storage.as_ref().ok_or(NarrowError{dim: 0,})?);
        let tensor =
            LongTensor::new_with_storage_nd(storage, offset, &new_size,
                                            &cur_stride);
        Ok(tensor)
    }
    #[cfg(feature = "safe")]
    fn narrow_on(&self, dim: usize, new_bound: Range<usize>)
     -> Result<LongTensor, NarrowError> {
        let (cur_shape, _) = self.shape();
        if new_bound.end <= cur_shape[dim] {
        } else { return Err(NarrowError{dim: dim,}) }
        let tensor =
            self.new_narrow(dim, new_bound.start,
                            new_bound.end - new_bound.start);
        Ok(tensor)
    }
    #[cfg(feature = "safe")]
    fn squeeze(&self) -> LongTensor {
        let mut new_ts = Self::new();
        new_ts.storage =
            match self.storage {
                Some(ref s) => Some(Rc::clone(s)),
                None => None,
            };
        unsafe {
            THLongTensor_squeeze(new_ts.tensor, self.tensor);
            let dim = THLongTensor_nDimension(new_ts.tensor);
            for i in 0..(dim - 1) {
                new_ts.size.push(THLongTensor_size(new_ts.tensor, i as c_int)
                                     as usize);
                new_ts.stride.push(THLongTensor_stride(new_ts.tensor,
                                                       i as c_int) as usize);
            }
            new_ts.size.push(THLongTensor_size(new_ts.tensor,
                                               dim - 1 as c_int) as usize);
            new_ts.stride.push(THLongTensor_stride(new_ts.tensor,
                                                   dim - 1 as c_int) as
                                   usize);
            new_ts
        }
    }
    #[cfg(feature = "safe")]
    fn view(&self, sizes: &[Option<usize>]) -> Result<LongTensor, ViewError> {
        let new_size = self.infer_size(sizes)?;
        let new_stride = self.compute_stride(&new_size)?;
        let offset = self.storage_offset();
        let mut storage =
            Rc::clone(self.storage.as_ref().ok_or(SizeInferError::ElementSizeMismatch)?);
        let ts =
            LongTensor::new_with_storage_nd(storage, offset, &new_size,
                                            &new_stride);
        Ok(ts)
    }
}
impl Tensor for LongTensor {
    type
    Datum
    =
    i64;
    type
    Storage
    =
    LongStorage;
}
impl <'a> Clone for LongTensor {
    #[cfg(feature = "safe")]
    fn clone(&self) -> LongTensor {
        unsafe {
            let clone = THLongTensor_newClone(self.tensor);
            let storage =
                Some(Rc::new(RefCell::new(LongStorage::from(THLongTensor_storage(clone)).forget())));
            let stride: Vec<usize> =
                (0..THLongTensor_nDimension(clone)).map(|i|
                                                            {
                                                                THLongTensor_stride(clone,
                                                                                    i
                                                                                        as
                                                                                        i32)
                                                                    as usize
                                                            }).collect();
            LongTensor{forget: false,
                       storage: storage,
                       tensor: clone,
                       size: self.size.to_owned(),
                       stride: stride,}
        }
    }
}
impl Drop for LongTensor {
    fn drop(&mut self) {
        unsafe { if !self.forget { THLongTensor_free(self.tensor); } }
    }
}
impl <'a> IntoIterator for &'a LongTensor {
    type
    Item
    =
    i64;
    type
    IntoIter
    =
    TensorIterator<'a, i64>;
    fn into_iter(self) -> Self::IntoIter {
        let (size, stride) = self.shape();
        TensorIterator::new(self.data(), size, stride)
    }
}
impl <'a> IntoIterator for &'a mut LongTensor {
    type
    Item
    =
    &'a mut i64;
    type
    IntoIter
    =
    TensorIterMut<'a, i64>;
    fn into_iter(self) -> Self::IntoIter {
        unsafe {
            let data = &mut *(self.data_mut() as *mut [i64]);
            let (size, stride) = self.shape();
            TensorIterMut::new(data, size, stride)
        }
    }
}
impl <'a> From<&'a [i64]> for LongTensor {
    #[cfg(feature = "safe")]
    fn from(slice: &'a [i64]) -> LongTensor {
        let mut tensor = LongTensor::new_with_size_1d(slice.len());
        tensor.storage().borrow_mut().unwrap().iter_mut().enumerate().for_each(|(i,
                                                                                 v)|
                                                                                   *v
                                                                                       =
                                                                                       slice[i]);
        tensor
    }
}
use storage::THShortStorage;
#[repr(C)]
pub struct THShortTensor;
#[link(name = "caffe2")]
extern "C" {
    fn THShortTensor_new() -> *mut THShortTensor;
    fn THShortTensor_newClone(org: *const THShortTensor)
     -> *mut THShortTensor;
    fn THShortTensor_newContiguous(org: *const THShortTensor)
     -> *mut THShortTensor;
    fn THShortTensor_newNarrow(org: *const THShortTensor, dim: c_int, i: i64,
                               size: i64) -> *mut THShortTensor;
    fn THShortTensor_newSelect(org: *const THShortTensor, dim: c_int,
                               idx: i64) -> *mut THShortTensor;
    fn THShortTensor_newTranspose(org: *const THShortTensor, dim_1: c_int,
                                  dim_2: c_int) -> *mut THShortTensor;
    fn THShortTensor_newUnfold(org: *const THShortTensor, dim: c_int,
                               size: i64, step: i64) -> *mut THShortTensor;
    fn THShortTensor_newWithTensor(org: *const THShortTensor)
     -> *mut THShortTensor;
    fn THShortTensor_newWithStorage1d(store: *mut THShortStorage,
                                      offset: usize, size: usize,
                                      stride: usize) -> *mut THShortTensor;
    fn THShortTensor_newWithStorage2d(store: *mut THShortStorage,
                                      offset: usize, size_1: usize,
                                      stride_1: usize, size_2: usize,
                                      stride_2: usize) -> *mut THShortTensor;
    fn THShortTensor_newWithStorage3d(store: *mut THShortStorage,
                                      offset: usize, size_1: usize,
                                      stride_1: usize, size_2: usize,
                                      stride_2: usize, size_3: usize,
                                      stride_3: usize) -> *mut THShortTensor;
    fn THShortTensor_newWithStorage4d(store: *mut THShortStorage,
                                      offset: usize, size_1: usize,
                                      stride_1: usize, size_2: usize,
                                      stride_2: usize, size_3: usize,
                                      stride_3: usize, size_4: usize,
                                      stride_4: usize) -> *mut THShortTensor;
    fn THShortTensor_newWithSize1d(size: i64) -> *mut THShortTensor;
    fn THShortTensor_newWithSize2d(size_1: i64, size_2: i64)
     -> *mut THShortTensor;
    fn THShortTensor_newWithSize3d(size_1: i64, size_2: i64, size_3: i64)
     -> *mut THShortTensor;
    fn THShortTensor_newWithSize4d(size_1: i64, size_2: i64, size_3: i64,
                                   size_4: i64) -> *mut THShortTensor;
    fn THShortTensor_free(tensor: *mut THShortTensor);
    fn THShortTensor_data(tensor: *mut THShortTensor) -> *mut i16;
    fn THShortTensor_desc(tensor: *mut THShortTensor) -> THDescBuff;
    fn THShortTensor_nDimension(tensor: *const THShortTensor) -> c_int;
    fn THShortTensor_isContiguous(tensor: *const THShortTensor) -> c_int;
    fn THShortTensor_get0d(tensor: *const THShortTensor) -> i16;
    fn THShortTensor_get1d(tensor: *const THShortTensor, i: i64) -> i16;
    fn THShortTensor_get2d(tensor: *const THShortTensor, i: i64, j: i64)
     -> i16;
    fn THShortTensor_get3d(tensor: *const THShortTensor, i: i64, j: i64,
                           k: i64) -> i16;
    fn THShortTensor_get4d(tensor: *const THShortTensor, i: i64, j: i64,
                           k: i64, l: i64) -> i16;
    fn THShortTensor_numel(tensor: *const THShortTensor) -> usize;
    fn THShortTensor_resize0d(tensor: *mut THShortTensor);
    fn THShortTensor_resize1d(tensor: *mut THShortTensor, size_1: i64);
    fn THShortTensor_resize2d(tensor: *mut THShortTensor, size_1: i64,
                              size_2: i64);
    fn THShortTensor_resize3d(tensor: *mut THShortTensor, size_1: i64,
                              size_2: i64, size_3: i64);
    fn THShortTensor_resize4d(tensor: *mut THShortTensor, size_1: i64,
                              size_2: i64, size_3: i64, size_4: i64);
    fn THShortTensor_resize5d(tensor: *mut THShortTensor, size_1: i64,
                              size_2: i64, size_3: i64, size_4: i64,
                              size_5: i64);
    fn THShortTensor_resizeNd(tensor: *mut THShortTensor, dim: c_int,
                              size: *const i64, stride: *const i64);
    fn THShortTensor_set0d(tensor: *const THShortTensor, v: i16);
    fn THShortTensor_set1d(tensor: *const THShortTensor, i: i64, v: i16);
    fn THShortTensor_set2d(tensor: *const THShortTensor, i: i64, j: i64,
                           v: i16);
    fn THShortTensor_set3d(tensor: *const THShortTensor, i: i64, j: i64,
                           k: i64, v: i16);
    fn THShortTensor_set4d(tensor: *const THShortTensor, i: i64, j: i64,
                           k: i64, l: i64, v: i16);
    fn THShortTensor_size(tensor: *const THShortTensor, dim: c_int) -> i64;
    fn THShortTensor_setStorageNd(tensor: *const THShortTensor,
                                  storage: THShortStorage, offset: i64,
                                  dim: c_int, size: *const i64,
                                  stride: *const i64);
    fn THShortTensor_storage(tensor: *mut THShortTensor)
     -> *mut THShortStorage;
    fn THShortTensor_storageOffset(tensor: *const THShortTensor) -> i64;
    fn THShortTensor_stride(tensor: *const THShortTensor, dim: c_int) -> i64;
    fn THShortTensor_squeeze(tensor: *mut THShortTensor,
                             src: *const THShortTensor);
}
pub struct ShortTensor {
    forget: bool,
    #[cfg(feature = "safe")]
    storage: Option<Rc<RefCell<ShortStorage>>>,
    tensor: *mut THShortTensor,
    size: Vec<usize>,
    stride: Vec<usize>,
}
impl ShortTensor {
    #[doc = r" Get short description of storage."]
    #[doc = r" This includes name of storage, size, and"]
    #[doc = r" sample data if it has more than 20 elements."]
    #[doc = r" If it has less than 20 elements, it'll display"]
    #[doc = r" every elements."]
    fn short_desc(&mut self) -> String {
        #[cfg(feature = "safe")]
        fn get_data(s: &ShortTensor) -> &ShortStorage {
            s.storage().as_ref().unwrap().borrow()
        }
        let size = self.size.as_slice();
        let stride = self.stride.as_slice();
        let data = get_data(self);
        let name = "ShortTensor";
        if size.iter().fold(0, |cum, v| cum + v) > 20 {
            ::alloc::fmt::format(::std::fmt::Arguments::new_v1(&["", ":size=",
                                                                 ":stride=",
                                                                 ":first(10)=",
                                                                 ":last(10)="],
                                                               &match (&name,
                                                                       &size,
                                                                       &stride,
                                                                       &&(&*data)[0..10],
                                                                       &&(&*data)[(data.len()
                                                                                       -
                                                                                       10)..data.len()])
                                                                    {
                                                                    (arg0,
                                                                     arg1,
                                                                     arg2,
                                                                     arg3,
                                                                     arg4) =>
                                                                    [::std::fmt::ArgumentV1::new(arg0,
                                                                                                 ::std::fmt::Display::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg1,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg2,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg3,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg4,
                                                                                                 ::std::fmt::Debug::fmt)],
                                                                }))
        } else {
            ::alloc::fmt::format(::std::fmt::Arguments::new_v1(&["", ":size=",
                                                                 ":stride=",
                                                                 ":data="],
                                                               &match (&name,
                                                                       &size,
                                                                       &stride,
                                                                       &data.iter())
                                                                    {
                                                                    (arg0,
                                                                     arg1,
                                                                     arg2,
                                                                     arg3) =>
                                                                    [::std::fmt::ArgumentV1::new(arg0,
                                                                                                 ::std::fmt::Display::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg1,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg2,
                                                                                                 ::std::fmt::Debug::fmt),
                                                                     ::std::fmt::ArgumentV1::new(arg3,
                                                                                                 ::std::fmt::Debug::fmt)],
                                                                }))
        }
    }
    #[doc = r" Alias for short_desc"]
    #[inline(always)]
    fn to_string(&mut self) -> String { self.short_desc() }
}
impl CreateOp<ShortStorage> for ShortTensor {
    type
    Datum
    =
    i16;
    #[cfg(feature = "safe")]
    fn new() -> ShortTensor {
        unsafe {
            ShortTensor{forget: false,
                        storage: None,
                        tensor: THShortTensor_new(),
                        size: Vec::new(),
                        stride: Vec::new(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_contiguous(&self) -> ShortTensor {
        unsafe {
            let cont = THShortTensor_newContiguous(self.tensor);
            let storage =
                Some(Rc::new(RefCell::new(ShortStorage::from(THShortTensor_storage(cont)))));
            let stride: Vec<usize> =
                (0..THShortTensor_nDimension(cont)).map(|i|
                                                            {
                                                                THShortTensor_stride(cont,
                                                                                     i
                                                                                         as
                                                                                         i32)
                                                                    as usize
                                                            }).collect();
            ShortTensor{forget: false,
                        storage: storage,
                        tensor: cont,
                        size: self.size.to_owned(),
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_narrow(&self, dim: usize, i: usize, size: usize) -> ShortTensor {
        unsafe {
            let tensor =
                THShortTensor_newNarrow(self.tensor, dim as c_int, i as i64,
                                        size as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THShortTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THShortTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     let cur_size =
                                         THShortTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            ShortTensor{forget: false,
                        storage: storage,
                        tensor: tensor,
                        size: size,
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_select(&self, dim: usize, i: usize) -> ShortTensor {
        unsafe {
            let tensor =
                THShortTensor_newSelect(self.tensor, dim as c_int, i as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THShortTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THShortTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     let cur_size =
                                         THShortTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            ShortTensor{forget: false,
                        storage: storage,
                        tensor: tensor,
                        size: size,
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_transpose(&self, dim_1: usize, dim_2: usize) -> ShortTensor {
        unsafe {
            let tensor =
                THShortTensor_newTranspose(self.tensor, dim_1 as c_int,
                                           dim_2 as c_int);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THShortTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THShortTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     let cur_size =
                                         THShortTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            ShortTensor{forget: false,
                        storage: storage,
                        tensor: tensor,
                        size: size,
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_unfold(&self, dim: usize, size: usize, step: usize)
     -> ShortTensor {
        unsafe {
            let tensor =
                THShortTensor_newUnfold(self.tensor, dim as c_int,
                                        size as i64, step as i64);
            let storage =
                match self.storage {
                    Some(ref s) => Some(Rc::clone(s)),
                    None => None,
                };
            let mut size = Vec::new();
            let dim = THShortTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THShortTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     let cur_size =
                                         THShortTensor_size(tensor, i as i32)
                                             as usize;
                                     size.push(cur_size);
                                     cur_stride
                                 }).collect();
            ShortTensor{forget: false,
                        storage: storage,
                        tensor: tensor,
                        size: size,
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_1d(store: Rc<RefCell<ShortStorage>>, offset: usize,
                           size: usize) -> ShortTensor {
        unsafe {
            let tensor =
                THShortTensor_newWithStorage1d(store.borrow_mut().storage(),
                                               offset, size, 1);
            ShortTensor{forget: false,
                        storage: Some(store),
                        tensor: tensor,
                        size: <[_]>::into_vec(box [size]),
                        stride: <[_]>::into_vec(box [1]),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_2d(store: Rc<RefCell<ShortStorage>>, offset: usize,
                           size: [usize; 2], stride: usize) -> ShortTensor {
        unsafe {
            let tensor =
                THShortTensor_newWithStorage2d(store.borrow_mut().storage(),
                                               offset, size[0], stride,
                                               size[1], 1);
            let dim = THShortTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THShortTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            ShortTensor{forget: false,
                        storage: Some(store),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_3d(store: Rc<RefCell<ShortStorage>>, offset: usize,
                           size: [usize; 3], stride: [usize; 2])
     -> ShortTensor {
        unsafe {
            let tensor =
                THShortTensor_newWithStorage3d(store.borrow_mut().storage(),
                                               offset, size[0], stride[0],
                                               size[1], stride[1], size[2],
                                               1);
            let dim = THShortTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THShortTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            ShortTensor{forget: false,
                        storage: Some(store),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_4d(store: Rc<RefCell<ShortStorage>>, offset: usize,
                           size: [usize; 4], stride: [usize; 3])
     -> ShortTensor {
        unsafe {
            let tensor =
                THShortTensor_newWithStorage4d(store.borrow_mut().storage(),
                                               offset, size[0], stride[0],
                                               size[1], stride[1], size[2],
                                               stride[2], size[3], 1);
            let dim = THShortTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THShortTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            let mut stride = stride.to_vec();
            stride.push(1);
            ShortTensor{forget: false,
                        storage: Some(store),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_storage_nd(store: Rc<RefCell<ShortStorage>>, offset: usize,
                           size: &[usize], stride: &[usize]) -> ShortTensor {
        if !(size.len() == stride.len() - 1 || size.len() == stride.len()) {
            {
                ::std::rt::begin_panic("Stride shall have either n - 1 elements or n elements where n = size.len()",
                                       &("tensor\\src\\lib.rs", 1300u32,
                                         1u32))
            }
        };
        let mut storage_len = 0;
        for i in 0..(stride.len() - 1) {
            let cur_len = stride[i] * size[i];
            if cur_len > storage_len { storage_len = cur_len; }
        }
        storage_len += size[size.len() - 1] - 1;
        let mut stride = stride.to_owned();
        if stride.len() == size.len() - 1 { stride.push(1); }
        unsafe {
            let tensor =
                THShortTensor_newWithStorage1d(store.borrow_mut().storage(),
                                               offset, storage_len, 1);
            THShortTensor_resizeNd(tensor, size.len() as i32,
                                   size.as_ptr() as *const i64,
                                   stride.as_ptr() as *const i64);
            let dim = THShortTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THShortTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            let stride = stride.to_vec();
            ShortTensor{forget: false,
                        storage: Some(store),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride,}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_1d(size: usize) -> ShortTensor {
        unsafe {
            let tensor = THShortTensor_newWithSize1d(size as i64);
            let stride = THShortTensor_stride(tensor, 0 as c_int) as usize;
            let storage =
                ShortStorage::from(THShortTensor_storage(tensor)).forget();
            ShortTensor{forget: false,
                        storage: Some(Rc::new(RefCell::new(storage))),
                        tensor: tensor,
                        size: <[_]>::into_vec(box [size]),
                        stride: <[_]>::into_vec(box [stride]),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_2d(size: [usize; 2]) -> ShortTensor {
        unsafe {
            let tensor =
                THShortTensor_newWithSize2d(size[0] as i64, size[1] as i64);
            let stride =
                [THShortTensor_stride(tensor, 0 as c_int) as usize,
                 THShortTensor_stride(tensor, 1 as c_int) as usize];
            let storage: ShortStorage = THShortTensor_storage(tensor).into();
            let dim = THShortTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THShortTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            ShortTensor{forget: false,
                        storage:
                            Some(Rc::new(RefCell::new(storage.forget()))),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride.to_vec(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_3d(size: [usize; 3]) -> ShortTensor {
        unsafe {
            let tensor =
                THShortTensor_newWithSize3d(size[0] as i64, size[1] as i64,
                                            size[2] as i64);
            let stride =
                [THShortTensor_stride(tensor, 0 as c_int) as usize,
                 THShortTensor_stride(tensor, 1 as c_int) as usize,
                 THShortTensor_stride(tensor, 2 as c_int) as usize];
            let storage =
                ShortStorage::from(THShortTensor_storage(tensor)).forget();
            let dim = THShortTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THShortTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            ShortTensor{forget: false,
                        storage: Some(Rc::new(RefCell::new(storage))),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride.to_vec(),}
        }
    }
    #[cfg(feature = "safe")]
    fn new_with_size_4d(size: [usize; 4]) -> ShortTensor {
        unsafe {
            let tensor =
                THShortTensor_newWithSize4d(size[0] as i64, size[1] as i64,
                                            size[2] as i64, size[3] as i64);
            let stride =
                [THShortTensor_stride(tensor, 0 as c_int) as usize,
                 THShortTensor_stride(tensor, 1 as c_int) as usize,
                 THShortTensor_stride(tensor, 2 as c_int) as usize,
                 THShortTensor_stride(tensor, 3 as c_int) as usize];
            let storage =
                ShortStorage::from(THShortTensor_storage(tensor)).forget();
            let dim = THShortTensor_nDimension(tensor);
            let stride: Vec<usize> =
                (0..dim).map(|i|
                                 {
                                     let cur_stride =
                                         THShortTensor_stride(tensor,
                                                              i as i32) as
                                             usize;
                                     cur_stride
                                 }).collect();
            ShortTensor{forget: false,
                        storage: Some(Rc::new(RefCell::new(storage))),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride.to_vec(),}
        }
    }
}
impl BasicManipulateOp<ShortStorage> for ShortTensor {
    type
    Datum
    =
    i16;
    fn desc(&self) -> String {
        unsafe { THShortTensor_desc(self.tensor).to_string() }
    }
    fn dimensions(&self) -> usize {
        unsafe { THShortTensor_nDimension(self.tensor) as usize }
    }
    fn get_0d(&self) -> i16 { unsafe { THShortTensor_get0d(self.tensor) } }
    fn get_1d(&self, i: usize) -> i16 {
        unsafe { THShortTensor_get1d(self.tensor, i as i64) }
    }
    fn get_2d(&self, i: [usize; 2]) -> i16 {
        unsafe { THShortTensor_get2d(self.tensor, i[0] as i64, i[1] as i64) }
    }
    fn get_3d(&self, i: [usize; 3]) -> i16 {
        unsafe {
            THShortTensor_get3d(self.tensor, i[0] as i64, i[1] as i64,
                                i[2] as i64)
        }
    }
    fn get_4d(&self, i: [usize; 4]) -> i16 {
        unsafe {
            THShortTensor_get4d(self.tensor, i[0] as i64, i[1] as i64,
                                i[2] as i64, i[3] as i64)
        }
    }
    fn iter(&self) -> TensorIterator<i16> { self.into_iter() }
    fn iter_mut(&mut self) -> TensorIterMut<i16> { self.into_iter() }
    fn is_contiguous(&self) -> bool {
        unsafe { THShortTensor_isContiguous(self.tensor) != 0 }
    }
    fn numel(&self) -> usize {
        unsafe { THShortTensor_numel(self.tensor) as usize }
    }
    #[cfg(feature = "safe")]
    fn resize_0d(&mut self) {
        unsafe {
            self.size.clear();
            self.stride.clear();
            THShortTensor_resize0d(self.tensor);
        }
    }
    #[cfg(feature = "safe")]
    fn resize_1d(&mut self, size: usize) {
        unsafe {
            self.size = <[_]>::into_vec(box [size]);
            self.stride = <[_]>::into_vec(box [1]);
            THShortTensor_resize1d(self.tensor, size as i64);
            self.stride =
                <[_]>::into_vec(box
                                    [THShortTensor_stride(self.tensor,
                                                          0 as c_int) as
                                         usize]);
        }
    }
    #[cfg(feature = "safe")]
    fn resize_2d(&mut self, size: [usize; 2]) {
        unsafe {
            self.size = size.to_vec();
            THShortTensor_resize2d(self.tensor, size[0] as i64,
                                   size[1] as i64);
            self.stride =
                [THShortTensor_stride(self.tensor, 0 as c_int) as usize,
                 THShortTensor_stride(self.tensor, 1 as c_int) as
                     usize].to_vec();
        }
    }
    #[cfg(feature = "safe")]
    fn resize_3d(&mut self, size: [usize; 3]) {
        unsafe {
            self.size = size.to_vec();
            THShortTensor_resize3d(self.tensor, size[0] as i64,
                                   size[1] as i64, size[2] as i64);
            self.stride =
                [THShortTensor_stride(self.tensor, 0 as c_int) as usize,
                 THShortTensor_stride(self.tensor, 1 as c_int) as usize,
                 THShortTensor_stride(self.tensor, 2 as c_int) as
                     usize].to_vec();
        }
    }
    #[cfg(feature = "safe")]
    fn resize_nd(&mut self, size: &[usize], stride: &[usize]) {
        {
            match (&(size.len() - 1), &(stride.len())) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        {
                            ::std::rt::begin_panic_fmt(&::std::fmt::Arguments::new_v1(&["assertion failed: `(left == right)`\n  left: `",
                                                                                        "`,\n right: `",
                                                                                        "`: "],
                                                                                      &match (&left_val,
                                                                                              &right_val,
                                                                                              &::std::fmt::Arguments::new_v1(&["Stride must have exactly ",
                                                                                                                               " elements"],
                                                                                                                             &match (&(size.len()
                                                                                                                                           -
                                                                                                                                           1),)
                                                                                                                                  {
                                                                                                                                  (arg0,)
                                                                                                                                  =>
                                                                                                                                  [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                                                               ::std::fmt::Display::fmt)],
                                                                                                                              }))
                                                                                           {
                                                                                           (arg0,
                                                                                            arg1,
                                                                                            arg2)
                                                                                           =>
                                                                                           [::std::fmt::ArgumentV1::new(arg0,
                                                                                                                        ::std::fmt::Debug::fmt),
                                                                                            ::std::fmt::ArgumentV1::new(arg1,
                                                                                                                        ::std::fmt::Debug::fmt),
                                                                                            ::std::fmt::ArgumentV1::new(arg2,
                                                                                                                        ::std::fmt::Display::fmt)],
                                                                                       }),
                                                       &("tensor\\src\\lib.rs",
                                                         1300u32, 1u32))
                        }
                    }
                }
            }
        };
        unsafe {
            self.size = size.to_owned();
            self.stride = stride.to_owned();
            self.stride.push(1);
            THShortTensor_resizeNd(self.tensor, size.len() as c_int,
                                   size.as_ptr() as *const i64,
                                   self.stride.as_ptr() as *const i64);
        }
    }
    fn set_0d(&mut self, v: i16) {
        unsafe { THShortTensor_set0d(self.tensor, v); }
    }
    fn set_1d(&mut self, i: usize, v: i16) {
        unsafe { THShortTensor_set1d(self.tensor, i as i64, v); }
    }
    fn set_2d(&mut self, i: [usize; 2], v: i16) {
        unsafe {
            THShortTensor_set2d(self.tensor, i[0] as i64, i[1] as i64, v);
        }
    }
    fn set_3d(&mut self, i: [usize; 3], v: i16) {
        unsafe {
            THShortTensor_set3d(self.tensor, i[0] as i64, i[1] as i64,
                                i[2] as i64, v);
        }
    }
    fn set_4d(&mut self, i: [usize; 4], v: i16) {
        unsafe {
            THShortTensor_set4d(self.tensor, i[0] as i64, i[1] as i64,
                                i[2] as i64, i[3] as i64, v);
        }
    }
    fn shape(&self) -> (&[usize], &[usize]) {
        (self.size.as_slice(), self.stride.as_slice())
    }
    fn size(&self, dim: usize) -> usize {
        if !(dim < self.size.len()) {
            {
                ::std::rt::begin_panic("assertion failed: dim < self.size.len()",
                                       &("tensor\\src\\lib.rs", 1300u32,
                                         1u32))
            }
        };
        unsafe { THShortTensor_size(self.tensor, dim as c_int) as usize }
    }
    #[cfg(feature = "safe")]
    fn storage(&mut self) -> &mut Option<Rc<RefCell<ShortStorage>>> {
        &mut self.storage
    }
    fn storage_offset(&self) -> usize {
        unsafe { THShortTensor_storageOffset(self.tensor) as usize }
    }
    fn stride(&self, dim: usize) -> usize {
        if !(dim < self.stride.len()) {
            {
                ::std::rt::begin_panic("assertion failed: dim < self.stride.len()",
                                       &("tensor\\src\\lib.rs", 1300u32,
                                         1u32))
            }
        };
        unsafe { THShortTensor_stride(self.tensor, dim as c_int) as usize }
    }
}
impl UtilityOp<ShortStorage> for ShortTensor { }
impl ViewOp<ShortTensor> for ShortTensor {
    #[cfg(feature = "safe")]
    fn narrow(&self, bound: &[Range<usize>])
     -> Result<ShortTensor, NarrowError> {
        let (cur_shape, cur_stride) = self.shape();
        let mut new_size = Vec::with_capacity(bound.len());
        let mut offset = self.storage_offset();
        for (((dim, u_bound), cur_stride), new_bound) in
            cur_shape.iter().enumerate().zip(cur_stride.iter()).zip(bound.iter())
            {
            if *u_bound < new_bound.end {
                return Err(NarrowError{dim: dim,})
            } else {
                new_size.push(new_bound.end - new_bound.start);
                offset += new_bound.start * *cur_stride;
            }
        }
        let storage =
            Rc::clone(self.storage.as_ref().ok_or(NarrowError{dim: 0,})?);
        let tensor =
            ShortTensor::new_with_storage_nd(storage, offset, &new_size,
                                             &cur_stride);
        Ok(tensor)
    }
    #[cfg(feature = "safe")]
    fn narrow_on(&self, dim: usize, new_bound: Range<usize>)
     -> Result<ShortTensor, NarrowError> {
        let (cur_shape, _) = self.shape();
        if new_bound.end <= cur_shape[dim] {
        } else { return Err(NarrowError{dim: dim,}) }
        let tensor =
            self.new_narrow(dim, new_bound.start,
                            new_bound.end - new_bound.start);
        Ok(tensor)
    }
    #[cfg(feature = "safe")]
    fn squeeze(&self) -> ShortTensor {
        let mut new_ts = Self::new();
        new_ts.storage =
            match self.storage {
                Some(ref s) => Some(Rc::clone(s)),
                None => None,
            };
        unsafe {
            THShortTensor_squeeze(new_ts.tensor, self.tensor);
            let dim = THShortTensor_nDimension(new_ts.tensor);
            for i in 0..(dim - 1) {
                new_ts.size.push(THShortTensor_size(new_ts.tensor, i as c_int)
                                     as usize);
                new_ts.stride.push(THShortTensor_stride(new_ts.tensor,
                                                        i as c_int) as usize);
            }
            new_ts.size.push(THShortTensor_size(new_ts.tensor,
                                                dim - 1 as c_int) as usize);
            new_ts.stride.push(THShortTensor_stride(new_ts.tensor,
                                                    dim - 1 as c_int) as
                                   usize);
            new_ts
        }
    }
    #[cfg(feature = "safe")]
    fn view(&self, sizes: &[Option<usize>])
     -> Result<ShortTensor, ViewError> {
        let new_size = self.infer_size(sizes)?;
        let new_stride = self.compute_stride(&new_size)?;
        let offset = self.storage_offset();
        let mut storage =
            Rc::clone(self.storage.as_ref().ok_or(SizeInferError::ElementSizeMismatch)?);
        let ts =
            ShortTensor::new_with_storage_nd(storage, offset, &new_size,
                                             &new_stride);
        Ok(ts)
    }
}
impl Tensor for ShortTensor {
    type
    Datum
    =
    i16;
    type
    Storage
    =
    ShortStorage;
}
impl <'a> Clone for ShortTensor {
    #[cfg(feature = "safe")]
    fn clone(&self) -> ShortTensor {
        unsafe {
            let clone = THShortTensor_newClone(self.tensor);
            let storage =
                Some(Rc::new(RefCell::new(ShortStorage::from(THShortTensor_storage(clone)).forget())));
            let stride: Vec<usize> =
                (0..THShortTensor_nDimension(clone)).map(|i|
                                                             {
                                                                 THShortTensor_stride(clone,
                                                                                      i
                                                                                          as
                                                                                          i32)
                                                                     as usize
                                                             }).collect();
            ShortTensor{forget: false,
                        storage: storage,
                        tensor: clone,
                        size: self.size.to_owned(),
                        stride: stride,}
        }
    }
}
impl Drop for ShortTensor {
    fn drop(&mut self) {
        unsafe { if !self.forget { THShortTensor_free(self.tensor); } }
    }
}
impl <'a> IntoIterator for &'a ShortTensor {
    type
    Item
    =
    i16;
    type
    IntoIter
    =
    TensorIterator<'a, i16>;
    fn into_iter(self) -> Self::IntoIter {
        let (size, stride) = self.shape();
        TensorIterator::new(self.data(), size, stride)
    }
}
impl <'a> IntoIterator for &'a mut ShortTensor {
    type
    Item
    =
    &'a mut i16;
    type
    IntoIter
    =
    TensorIterMut<'a, i16>;
    fn into_iter(self) -> Self::IntoIter {
        unsafe {
            let data = &mut *(self.data_mut() as *mut [i16]);
            let (size, stride) = self.shape();
            TensorIterMut::new(data, size, stride)
        }
    }
}
impl <'a> From<&'a [i16]> for ShortTensor {
    #[cfg(feature = "safe")]
    fn from(slice: &'a [i16]) -> ShortTensor {
        let mut tensor = ShortTensor::new_with_size_1d(slice.len());
        tensor.storage().borrow_mut().unwrap().iter_mut().enumerate().for_each(|(i,
                                                                                 v)|
                                                                                   *v
                                                                                       =
                                                                                       slice[i]);
        tensor
    }
}
impl <'a> From<&'a ByteTensor> for CharTensor {
    /// Perform type casting from ByteTensor to CharTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a ByteTensor) -> CharTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = CharStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest = *src as i8);
        return CharTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a ByteTensor> for FloatTensor {
    /// Perform type casting from ByteTensor to FloatTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a ByteTensor) -> FloatTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = FloatStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as f32);
        return FloatTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                0, size, stride);
    }
}
impl <'a> From<&'a ByteTensor> for DoubleTensor {
    /// Perform type casting from ByteTensor to DoubleTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a ByteTensor) -> DoubleTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = DoubleStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as f64);
        return DoubleTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                 0, size, stride);
    }
}
impl <'a> From<&'a ByteTensor> for IntTensor {
    /// Perform type casting from ByteTensor to IntTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a ByteTensor) -> IntTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = IntStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i32);
        return IntTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                              0, size, stride);
    }
}
impl <'a> From<&'a ByteTensor> for LongTensor {
    /// Perform type casting from ByteTensor to LongTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a ByteTensor) -> LongTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = LongStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i64);
        return LongTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a ByteTensor> for ShortTensor {
    /// Perform type casting from ByteTensor to ShortTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a ByteTensor) -> ShortTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = ShortStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i16);
        return ShortTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                0, size, stride);
    }
}
impl <'a> From<&'a CharTensor> for ByteTensor {
    /// Perform type casting from CharTensor to ByteTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a CharTensor) -> ByteTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = ByteStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest = *src as u8);
        return ByteTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a CharTensor> for FloatTensor {
    /// Perform type casting from CharTensor to FloatTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a CharTensor) -> FloatTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = FloatStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as f32);
        return FloatTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                0, size, stride);
    }
}
impl <'a> From<&'a CharTensor> for DoubleTensor {
    /// Perform type casting from CharTensor to DoubleTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a CharTensor) -> DoubleTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = DoubleStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as f64);
        return DoubleTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                 0, size, stride);
    }
}
impl <'a> From<&'a CharTensor> for IntTensor {
    /// Perform type casting from CharTensor to IntTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a CharTensor) -> IntTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = IntStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i32);
        return IntTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                              0, size, stride);
    }
}
impl <'a> From<&'a CharTensor> for LongTensor {
    /// Perform type casting from CharTensor to LongTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a CharTensor) -> LongTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = LongStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i64);
        return LongTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a CharTensor> for ShortTensor {
    /// Perform type casting from CharTensor to ShortTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a CharTensor) -> ShortTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = ShortStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i16);
        return ShortTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                0, size, stride);
    }
}
impl <'a> From<&'a FloatTensor> for ByteTensor {
    /// Perform type casting from FloatTensor to ByteTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a FloatTensor) -> ByteTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = ByteStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest = *src as u8);
        return ByteTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a FloatTensor> for CharTensor {
    /// Perform type casting from FloatTensor to CharTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a FloatTensor) -> CharTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = CharStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest = *src as i8);
        return CharTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a FloatTensor> for DoubleTensor {
    /// Perform type casting from FloatTensor to DoubleTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a FloatTensor) -> DoubleTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = DoubleStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as f64);
        return DoubleTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                 0, size, stride);
    }
}
impl <'a> From<&'a FloatTensor> for IntTensor {
    /// Perform type casting from FloatTensor to IntTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a FloatTensor) -> IntTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = IntStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i32);
        return IntTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                              0, size, stride);
    }
}
impl <'a> From<&'a FloatTensor> for LongTensor {
    /// Perform type casting from FloatTensor to LongTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a FloatTensor) -> LongTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = LongStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i64);
        return LongTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a FloatTensor> for ShortTensor {
    /// Perform type casting from FloatTensor to ShortTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a FloatTensor) -> ShortTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = ShortStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i16);
        return ShortTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                0, size, stride);
    }
}
impl <'a> From<&'a DoubleTensor> for ByteTensor {
    /// Perform type casting from DoubleTensor to ByteTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a DoubleTensor) -> ByteTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = ByteStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest = *src as u8);
        return ByteTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a DoubleTensor> for CharTensor {
    /// Perform type casting from DoubleTensor to CharTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a DoubleTensor) -> CharTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = CharStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest = *src as i8);
        return CharTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a DoubleTensor> for FloatTensor {
    /// Perform type casting from DoubleTensor to FloatTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a DoubleTensor) -> FloatTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = FloatStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as f32);
        return FloatTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                0, size, stride);
    }
}
impl <'a> From<&'a DoubleTensor> for IntTensor {
    /// Perform type casting from DoubleTensor to IntTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a DoubleTensor) -> IntTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = IntStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i32);
        return IntTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                              0, size, stride);
    }
}
impl <'a> From<&'a DoubleTensor> for LongTensor {
    /// Perform type casting from DoubleTensor to LongTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a DoubleTensor) -> LongTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = LongStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i64);
        return LongTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a DoubleTensor> for ShortTensor {
    /// Perform type casting from DoubleTensor to ShortTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a DoubleTensor) -> ShortTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = ShortStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i16);
        return ShortTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                0, size, stride);
    }
}
impl <'a> From<&'a IntTensor> for ByteTensor {
    /// Perform type casting from IntTensor to ByteTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a IntTensor) -> ByteTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = ByteStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest = *src as u8);
        return ByteTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a IntTensor> for CharTensor {
    /// Perform type casting from IntTensor to CharTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a IntTensor) -> CharTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = CharStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest = *src as i8);
        return CharTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a IntTensor> for FloatTensor {
    /// Perform type casting from IntTensor to FloatTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a IntTensor) -> FloatTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = FloatStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as f32);
        return FloatTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                0, size, stride);
    }
}
impl <'a> From<&'a IntTensor> for DoubleTensor {
    /// Perform type casting from IntTensor to DoubleTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a IntTensor) -> DoubleTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = DoubleStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as f64);
        return DoubleTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                 0, size, stride);
    }
}
impl <'a> From<&'a IntTensor> for LongTensor {
    /// Perform type casting from IntTensor to LongTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a IntTensor) -> LongTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = LongStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i64);
        return LongTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a IntTensor> for ShortTensor {
    /// Perform type casting from IntTensor to ShortTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a IntTensor) -> ShortTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = ShortStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i16);
        return ShortTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                0, size, stride);
    }
}
impl <'a> From<&'a LongTensor> for ByteTensor {
    /// Perform type casting from LongTensor to ByteTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a LongTensor) -> ByteTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = ByteStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest = *src as u8);
        return ByteTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a LongTensor> for CharTensor {
    /// Perform type casting from LongTensor to CharTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a LongTensor) -> CharTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = CharStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest = *src as i8);
        return CharTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a LongTensor> for FloatTensor {
    /// Perform type casting from LongTensor to FloatTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a LongTensor) -> FloatTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = FloatStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as f32);
        return FloatTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                0, size, stride);
    }
}
impl <'a> From<&'a LongTensor> for DoubleTensor {
    /// Perform type casting from LongTensor to DoubleTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a LongTensor) -> DoubleTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = DoubleStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as f64);
        return DoubleTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                 0, size, stride);
    }
}
impl <'a> From<&'a LongTensor> for IntTensor {
    /// Perform type casting from LongTensor to IntTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a LongTensor) -> IntTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = IntStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i32);
        return IntTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                              0, size, stride);
    }
}
impl <'a> From<&'a LongTensor> for ShortTensor {
    /// Perform type casting from LongTensor to ShortTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a LongTensor) -> ShortTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = ShortStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i16);
        return ShortTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                0, size, stride);
    }
}
impl <'a> From<&'a ShortTensor> for ByteTensor {
    /// Perform type casting from ShortTensor to ByteTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a ShortTensor) -> ByteTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = ByteStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest = *src as u8);
        return ByteTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a ShortTensor> for CharTensor {
    /// Perform type casting from ShortTensor to CharTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a ShortTensor) -> CharTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = CharStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest = *src as i8);
        return CharTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
impl <'a> From<&'a ShortTensor> for FloatTensor {
    /// Perform type casting from ShortTensor to FloatTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a ShortTensor) -> FloatTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = FloatStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as f32);
        return FloatTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                0, size, stride);
    }
}
impl <'a> From<&'a ShortTensor> for DoubleTensor {
    /// Perform type casting from ShortTensor to DoubleTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a ShortTensor) -> DoubleTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = DoubleStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as f64);
        return DoubleTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                                 0, size, stride);
    }
}
impl <'a> From<&'a ShortTensor> for IntTensor {
    /// Perform type casting from ShortTensor to IntTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a ShortTensor) -> IntTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = IntStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i32);
        return IntTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                              0, size, stride);
    }
}
impl <'a> From<&'a ShortTensor> for LongTensor {
    /// Perform type casting from ShortTensor to LongTensor.
    /// This is done by deep cloning on entire storage while also
    /// casting each element in storage from `u8` to `i8` at the same time.
    /// The return tensor is completely independent from original tensor.
    fn from(src: &'a ShortTensor) -> LongTensor {
        let (size, stride) = src.shape();
        let src_data = src.storage.as_ref().unwrap().borrow();
        let mut storage = LongStorage::new_with_size(src_data.len());
        let data = storage.data_mut();
        data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)|
                                                          *dest =
                                                              *src as i64);
        return LongTensor::new_with_storage_nd(Rc::new(RefCell::new(storage)),
                                               0, size, stride);
    }
}
