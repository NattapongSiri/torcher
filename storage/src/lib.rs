//! Rust abstraction over underlying Caffe2 tensor storage.
//! These structs implement Deref and DerefMut.
//! When deref, it return a slice that map directly
//! to internal data of Caffe2 tensor.
//! 
//! These tensor storages when constructed, need to
//! be carefully share. The default behavior is when
//! it no longer in the scope, it'll call free function
//! to return used memory. This should only be called once.
//! So the longest one shall be the one responsible for it.
//! All other shared storage shall called method 
//! [forget](trait.TensorStorage.html#tymethod.forget).
//! Failure to do so may cause segmentation fault.
//! 
//! Currently, it support following tensor storage:
//! - Single byte storage
//!     - [ByteStorage](struct.ByteStorage.html) - stores u8
//!     - [CharStorage](struct.CharStorage.html) - stores i8
//! - Floating point storage
//!     - [DoubleStorage](struct.DoubleStorage.html) - stores f64
//!     - [FloatStorage](struct.FloatStorage.html) - stores f32
//! - Integer storage
//!     - [IntStorage](struct.IntStorage.html) - stores i32
//!     - [LongStorage](struct.LongStorage.html) - stores i64
//!     - [ShortStorage](struct.ShortStorage.html) - stores i16
//! It has no HalfStorage which suppose to stores f16 because
//! Rust has no f16 type in core/std Rust.
//! It has no support for i128 as well because Caffe2 has no
//! such tensor storage.
//! 
//! All other structs are placeholder for mapping with Caffe2 
//! tensor storage pointer.
//! 
//! All tensor storage type defined here implement Deref and DerefMut.
//! It'll deref into underlying storage as Slice and SliceMut respectively.
extern crate storage_derive;

use std::ops::{Deref, DerefMut};
use storage_derive::{TorchStorage};

/// All functions operating on storage.
/// The storage is the root source of data for every tensor.
pub trait TensorStorage<T> : Drop {
    /// Construct new empty Tensor storage
    fn new() -> Self;
    /// Construct new Tensor storage and allocate
    /// fixed size on memory
    fn new_with_size(size: usize) -> Self;

    /// Return a slice of actual memory behind this storage.
    /// For fastest performance, it's recommend to use mutate data by using this function
    /// instead.
    fn data(&self) -> &[T];

    /// Return a mutable slice of actual memory behind
    /// this storage.
    /// For fastest performance, it's recommend to use mutate data by using this function
    /// instead.
    fn data_mut(&mut self) -> &mut [T];

    /// Consume this storage without freeing storage in 
    /// memory. This function is design to be used in FFI case
    /// where the underlying storage will be managed outside 
    /// of Rust.
    /// 
    /// # Safety
    /// Unless you know what you are doing, this will cause
    /// a memory leak. 
    unsafe fn forget(self) -> Self;

    /// Fill entire storage with given value.
    fn fill(&mut self, value: T);

    /// Alias of size method.
    /// This method is provided because most of Rust 
    /// code use `len` method to check for size.
    fn len(&self) -> usize {
        self.size()
    }

    /// Resize underlying allocated storage in memory.
    fn resize(&mut self, size: usize);

    /// ???
    fn retain(&mut self);

    /// Return current size of this storage
    fn size(&self) -> usize;

    /// Swap storage with another storage
    fn swap(&mut self, with : Self);
}

// Floating point storage
#[TorchStorage(f32)]
pub struct FloatStorage;

#[TorchStorage(f64)]
pub struct DoubleStorage;

// Byte storage
#[TorchStorage(u8)]
pub struct ByteStorage;

// Char storage
#[TorchStorage(i8)]
pub struct CharStorage;

// Integer storage
#[TorchStorage(i16)]
pub struct ShortStorage;

#[TorchStorage(i32)]
pub struct IntStorage;

#[TorchStorage(i64)]
pub struct LongStorage;


#[cfg(test)]
mod tests;