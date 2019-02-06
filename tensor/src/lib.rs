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

use core::ops::Range;
use common::THDescBuff;
use std::cmp;
use std::iter::Iterator;
use std::ops::{Deref, DerefMut};
use std::os::raw::{c_int};
use storage::{ByteStorage, CharStorage, DoubleStorage, FloatStorage, IntStorage, LongStorage, ShortStorage, TensorStorage};
use tensor_derive::TorchTensor;

/// Basic tensor operation for simple data manipulation.
/// This includes data read/write operation and tensor shape
/// related operation.
pub trait BasicManipulateOp<S: TensorStorage> {
    type Datum;

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
    fn data(&self) -> &[Self::Datum];
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
    fn data_mut(&mut self) -> &mut [Self::Datum];
    /// Just a wrapper to Caffe2 function.
    /// It currently only print size of tensor.
    fn desc(&self) -> String;
    /// Get total number of dimension this tensor is representing.
    fn dimensions(&self) -> usize;
    /// Get scalar value out of this scalar tensor.
    fn get_0d(&self) -> Self::Datum;
    /// Get scalar value from given index of tensor.
    fn get_1d(&self, i: usize) -> Self::Datum;
    /// Get scalar value from given indices of 2d tensor.
    fn get_2d(&self, i: [usize; 2]) -> Self::Datum;
    /// Get scalar value from given indices of 3d tensor
    fn get_3d(&self, i: [usize; 3]) -> Self::Datum;
    /// Get scalar value from given indices of 4d tensor
    fn get_4d(&self, i: [usize; 4]) -> Self::Datum;
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
    fn shape(&self) -> (&[usize], &[usize]);
    /// Return size of given dimension of this tensor
    fn size(&self, dim: usize) -> usize;
    /// Return underlying storage.
    /// If it's empty tensor, it may be None
    fn storage(&mut self) -> &mut Option<S>;
    /// Return storage offset of this tensor
    fn storage_offset(&self) -> usize;
    /// Return stride of given dimension of this tensor
    fn stride(&self, dim: usize) -> usize;
}

/// Tensor create operation.
pub trait CreateOp<S: TensorStorage> {
    type Datum;

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
    /// The underlying storage will be free when this tensor is drop
    fn new_with_storage_1d(store: S, offset: usize, size: usize) -> Self;
    /// Consume storage and associate it with new tensor.
    /// It map directly with Caffe2 function that responsible to do the similar task.
    /// 
    /// The underlying storage will be free when this tensor is drop
    fn new_with_storage_2d(store: S, offset: usize, size: [usize; 2], stride: usize) -> Self;
    /// Consume storage and associate it with new tensor.
    /// It map directly with Caffe2 function that responsible to do the similar task.
    /// 
    /// The underlying storage will be free when this tensor is drop
    fn new_with_storage_3d(store: S, offset: usize, size: [usize; 3], stride: [usize; 2]) -> Self;
    /// Consume storage and associate it with new tensor.
    /// It map directly with Caffe2 function that responsible to do the similar task.
    /// 
    /// The underlying storage will be free when this tensor is drop
    fn new_with_storage_4d(store: S, offset: usize, size: [usize; 4], stride: [usize; 3]) -> Self;
    /// Consume storage and associate it with new tensor.
    /// It map directly with Caffe2 function that responsible to do the similar task.
    /// 
    /// The underlying storage will be free when this tensor is drop
    fn new_with_storage_nd(store: S, offset: usize, size: &[usize], stride: &[usize]) -> Self;
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

    /// Leak this tensor. It'll not clean up memory occupy by this
    /// tensor when it goes out of scope.
    /// 
    /// # Safety
    /// It cause memory leak if all instant that use underlying
    /// Caffe2 tensors are mark with forget.
    /// There should be one and only one tensor that doesn't forget
    /// it underlying Caffe2 tensor.
    unsafe fn forget(self) -> Self;
}

/// A trait that all concrete tensor derivative need to implemented
pub trait Tensor: 
BasicManipulateOp<<Self as Tensor>::Storage, Datum=<Self as Tensor>::Datum> + 
CreateOp<<Self as Tensor>::Storage, Datum=<Self as Tensor>::Datum> +
ViewOp<Self>
where Self: Sized
{
    type Datum;
    type Storage: TensorStorage<Datum=<Self as Tensor>::Datum>;
}

trait UtilityOp<S>: BasicManipulateOp<S>
where S: TensorStorage
{
    /// This function will convert a None entry in size into numeric.
    /// It'll return Err if the there's more than one None inside `sizes`.
    /// It'll return Err if the new sizes and old sizes have different number
    /// of elements and there's no single `None` element.
    /// It'll also panic if all the specified size use up all elements of 
    /// the tensor and there's one `None` element.
    /// It'll panic if one of the size is 0.
    /// A direct port from Caffe2 [infer_size function](https://github.com/pytorch/pytorch/blob/21907b6ba2a37afff8f23111b4f98a83fe2b093d/aten/src/ATen/InferSize.h#L12) into Rust
    fn infer_size(&self, sizes: &[Option<usize>]) -> Result<Vec<usize>, SizeInferError> {
        let numel = self.numel();
        let mut res : Vec<usize> = Vec::with_capacity(sizes.len());
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
            if let Some(dim) = infer_dim {
                res[dim] = numel / new_size;
            } 

            return Ok(res);
        }

        return Err(SizeInferError::ElementSizeMismatch);
    }

    /// Compute stride based on current size and stride for new given size.
    /// 
    /// This method is a direct port from Caffe2 API
    /// [compute_stride function](https://github.com/pytorch/pytorch/blob/21907b6ba2a37afff8f23111b4f98a83fe2b093d/aten/src/TH/THTensor.cpp#L93)
    fn compute_stride(&self, sizes: &[usize]) -> Result<Vec<usize>, StrideComputeError> {
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
                    new_stride.push(cmp::max(sizes[view_d + 1], 1) * new_stride[sizes.len() - view_d - 1]);
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
                (old_size[tensor_d - 1] != 1 && old_stride[tensor_d - 1] != tensor_numel * chunk_base_stride) {
                while view_d > 0 && (view_numel < tensor_numel || sizes[view_d] == 1) {
                    new_stride.push(view_numel * chunk_base_stride);
                    view_numel *= sizes[view_d];
                    view_d -= 1;
                }

                if view_d == 0 {
                    new_stride.push(view_numel * chunk_base_stride);
                    view_numel *= sizes[view_d];
                }
                if view_numel != tensor_numel {
                    return Err(StrideComputeError {});
                }
                if tensor_d > 0 {
                    chunk_base_stride = old_stride[tensor_d - 1];
                    tensor_numel = 1;
                    view_numel = 1;
                }
            }
        }
        
        if view_d != 0 {
            return Err(StrideComputeError {});
        }

        new_stride.reverse();

        return Ok(new_stride);
    }
}

/// View related operation
pub trait ViewOp<T: Tensor> {
    /// Return an original tensor before any view is applied.
    /// If it is called on original tensor, it return itself.
    fn original(self) -> T;
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
    fn narrow(self, bound: &[Range<usize>]) -> Result<TensorView<T>, NarrowError>;
    /// Apply narrow on specific dimension and return a narrowed view on
    /// given tensor along with the original tensor.
    fn narrow_on(self, dim: usize, new_bound: Range<usize>) -> Result<TensorView<T>, NarrowError>;

    /// Perform tensor squeeze. It'll flatten any dimension
    /// that have size 1 and return the new squeezed TensorView
    fn squeeze(self) -> TensorView<T>;
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
    fn view(self, bound: &[Option<usize>]) -> Result<TensorView<T>, ViewError>;

}

#[derive(Debug)]
pub enum SizeInferError {
    /// Multiple None elements given to infer_size.
    /// It need to have at most 1 None element.
    MultipleUnsizedError,
    /// The number of elements of tensor isn't compatible with
    /// given size. The new size may either too small or too large.
    /// The new size must use all elements of the tensor.
    ElementSizeMismatch
}

/// Cannot compute stride based on given size.
/// Either one of new view span over noncontiguous dimension
/// or view size exceed number of element.
#[derive(Debug)]
pub struct StrideComputeError {}

/// Narrow operation fail.
/// Potential cause is one of range end is out of bound
#[derive(Debug)]
pub struct NarrowError {
    dim: usize
}

/// Cannot make a view from given sizes
#[derive(Debug)]
pub enum ViewError {
    /// One of size infer related error occur
    SizeErr(SizeInferError),
    /// Stride computation error
    StrideErr(StrideComputeError),
}
impl From<SizeInferError> for ViewError {
    fn from(err: SizeInferError) -> Self {
        ViewError::SizeErr(err)
    }
}

impl From<StrideComputeError> for ViewError {
    fn from(err: StrideComputeError) -> Self {
        ViewError::StrideErr(err)
    }
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

/// A view wrapped Tensor. It provide safety abstraction when
/// viewing tensor. 
pub struct TensorView<T> where T: Tensor {
    original: T,
    view: T
}

impl<T> TensorView<T> where T: Tensor {
    /// Consume the view and return the original tensor
    pub fn original(self) -> T {
        self.original
    }
}

/// Simple data manipulation on the tensor in view.
impl<T> BasicManipulateOp<<T as Tensor>::Storage> for TensorView<T> where T: Tensor {
    type Datum = <T as Tensor>::Datum;

    fn data(&self) -> &[Self::Datum] {
        self.view.data()
    }
    fn data_mut(&mut self) -> &mut [Self::Datum] {
        self.view.data_mut()
    }
    fn desc(&self) -> String {
        self.view.desc()
    }
    fn dimensions(&self) -> usize {
        self.view.dimensions()
    }
    fn get_0d(&self) -> Self::Datum {
        self.view.get_0d()
    }
    fn get_1d(&self, i: usize) -> Self::Datum {
        self.view.get_1d(i)
    }
    fn get_2d(&self, i: [usize; 2]) -> Self::Datum {
        self.view.get_2d(i)
    }
    fn get_3d(&self, i: [usize; 3]) -> Self::Datum {
        self.view.get_3d(i)
    }
    fn get_4d(&self, i: [usize; 4]) -> Self::Datum {
        self.view.get_4d(i)
    }
    fn is_contiguous(&self) -> bool {
        self.view.is_contiguous()
    }
    fn numel(&self) -> usize {
        self.view.numel()
    }
    fn resize_0d(&mut self) {
        self.view.resize_0d();
    }
    fn resize_1d(&mut self, size: usize) {
        self.view.resize_1d(size);
    }
    fn resize_2d(&mut self, size: [usize; 2]) {
        self.view.resize_2d(size);
    }
    fn resize_3d(&mut self, size: [usize; 3]) {
        self.view.resize_3d(size);
    }
    fn resize_nd(&mut self, size: &[usize], stride: &[usize]) {
        self.view.resize_nd(size, stride);
    }
    fn set_0d(&mut self, v: Self::Datum) {
        self.view.set_0d(v);
    }
    fn set_1d(&mut self, i: usize, v: Self::Datum) {
        self.view.set_1d(i, v);
    }
    fn set_2d(&mut self, i: [usize; 2], v: Self::Datum) {
        self.view.set_2d(i, v);
    }
    fn set_3d(&mut self, i: [usize; 3], v: Self::Datum) {
        self.view.set_3d(i, v);
    }
    fn set_4d(&mut self, i: [usize; 4], v: Self::Datum) {
        self.view.set_4d(i, v);
    }
    fn shape(&self) -> (&[usize], &[usize]) {
        self.view.shape()
    }
    fn size(&self, dim: usize) -> usize {
        self.view.size(dim)
    }
    fn storage(&mut self) -> &mut Option<<T as Tensor>::Storage> {
        self.view.storage()
    }
    fn storage_offset(&self) -> usize {
        self.view.storage_offset()
    }
    fn stride(&self, dim: usize) -> usize {
        self.view.stride(dim)
    }
}

impl<T> Deref for TensorView<T> where T: Tensor {
    type Target=T;

    fn deref(&self) -> &Self::Target {
        &self.view
    }
}

impl<T> ViewOp<T> for TensorView<T>
where T: Tensor
{
    /// Return an original tensor before any view is applied.
    fn original(self) -> T {
        self.original
    }

    /// Perform narrow down the view. 
    /// __Note:__ It'll save original tensor along
    /// with new view. The original tensor is the tensor before
    /// first view operation is perform.
    fn narrow(self, bound: &[Range<usize>]) -> Result<TensorView<T>, NarrowError> {
        Ok(
            Self {
                original: self.original,
                view: self.view.narrow(bound)?.view
            }
        )
    }

    /// Perform narrow down the view on given dimension. 
    /// __Note:__ It'll save original tensor along
    /// with new view. The original tensor is the tensor before
    /// first view operation is perform.
    fn narrow_on(self, dim: usize, bound: Range<usize>) -> Result<TensorView<T>, NarrowError> {
        Ok(
            Self {
                original: self.original,
                view: self.view.narrow_on(dim, bound)?.view
            }
        )
    }

    /// Perform tensor squeeze. It'll flatten any dimension
    /// that have size 1 and return the new squeezed TensorView
    /// __Note:__ It'll save original tensor along
    /// with new view. The original tensor is the tensor before
    /// first view operation is perform.
    fn squeeze(self) -> TensorView<T> {
        Self {
            original: self.original,
            view: self.view.squeeze().view 
        }
    }

    /// Perform sub-view on this view. The new sub-view will not store
    /// current view as original but use the original tensor as original.
    fn view(self, sizes: &[Option<usize>]) -> Result<TensorView<T>, ViewError> {
        Ok(
            Self {
                original: self.original,
                view: self.view.view(sizes)?.view
            }
        )
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

include!(concat!(env!("OUT_DIR"), "/conv.rs"));

#[cfg(test)]
mod tests;