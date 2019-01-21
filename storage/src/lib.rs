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

    /// Return a mutable slice of actual memory behind
    /// this storage.
    /// For fastest performance, it's recommend to use mutate data by using this function
    /// instead.
    fn data<'a>(&'a mut self) -> &'a mut [T];

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
mod tests {
    use super::*;

    #[test]
    fn float_construct_destroy() {
        FloatStorage::new();
    }

    #[test]
    fn double_construct_destroy() {
        DoubleStorage::new();
    }

    #[test]
    fn double_from() {
        let fs = FloatStorage::new_with_size(5);
        unsafe {
            // auto convert using From trait
            println!("Attempt to convert from native c store");
            let native_store = fs.storage();
            let another_fs : FloatStorage = native_store.into();
            another_fs.forget(); // or it will crash because underlying storage got free twice
            println!("Conversion succeed.");
        }
    }

    #[test]
    fn float_fill_display() {
        let mut fs = FloatStorage::new_with_size(5);
        fs.fill(1f32);
        assert_eq!("FloatStorage:size=5:data=[1.0, 1.0, 1.0, 1.0, 1.0]", fs.to_string());
    }

    #[test]
    fn double_fill_display() {
        let mut fs = DoubleStorage::new_with_size(5);
        fs.fill(1.0f64);
        assert_eq!("DoubleStorage:size=5:data=[1.0, 1.0, 1.0, 1.0, 1.0]", fs.to_string());
    }

    #[test]
    fn double_fill_display_large() {
        let mut fs = DoubleStorage::new_with_size(50);
        fs.fill(2.0f64);
        assert_eq!("DoubleStorage:size=50:first(10)=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]:last(10)=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]", fs.to_string());
    }

    #[test]
    fn float_mutate_data() {
        let mut fs = FloatStorage::new_with_size(32);
        let mut validator = Vec::with_capacity(32);

        fs.data().iter_mut().enumerate().for_each(|(i, d)| {
            *d = i as f32;
            validator.push(i as f32);
        });

        assert_eq!(fs.data(), validator.as_slice());
    }

    #[test]
    fn double_mutate_data() {
        let mut fs = DoubleStorage::new_with_size(32);
        let mut validator = Vec::with_capacity(32);

        fs.data().iter_mut().enumerate().for_each(|(i, d)| {
            *d = i as f64;
            validator.push(i as f64);
        });

        assert_eq!(fs.data(), validator.as_slice());
    }

    #[test]
    fn float_resize_mutate_data() {
        let mut fs = FloatStorage::new_with_size(16);
        fs.resize(32);
        let mut validator = Vec::with_capacity(32);

        fs.data().iter_mut().enumerate().for_each(|(i, d)| {
            *d = i as f32;
            validator.push(i as f32);
        });

        assert_eq!(fs.data(), validator.as_slice());
    }

    #[test]
    fn double_resize_mutate_data() {
        let mut fs = DoubleStorage::new_with_size(16);
        fs.resize(32);
        let mut validator = Vec::with_capacity(32);

        fs.data().iter_mut().enumerate().for_each(|(i, d)| {
            *d = i as f64;
            validator.push(i as f64);
        });

        assert_eq!(fs.data(), validator.as_slice());
    }

    #[test]
    fn float_mutate_resize_data() {
        let mut fs = FloatStorage::new_with_size(32);
        let mut validator = Vec::with_capacity(32);

        fs.data().iter_mut().enumerate().for_each(|(i, d)| {
            *d = i as f32;
            validator.push(i as f32);
        });

        fs.resize(16);

        assert_eq!(fs.data(), &validator[0..16]);
    }

    #[test]
    fn double_mutate_resize_data() {
        let mut fs = DoubleStorage::new_with_size(32);
        let mut validator = Vec::with_capacity(32);

        fs.data().iter_mut().enumerate().for_each(|(i, d)| {
            *d = i as f64;
            validator.push(i as f64);
        });

        fs.resize(16);

        assert_eq!(fs.data(), &validator[0..16]);
    }

    #[test]
    fn float_index() {
        let mut fs = FloatStorage::new_with_size(32);
        let mut validator = Vec::with_capacity(32);

        fs.data().iter_mut().enumerate().for_each(|(i, d)| {
            *d = i as f32;
            validator.push(i as f32);
        });

        assert_eq!(31f32, fs[fs.len() - 1]);
    }

    #[test]
    fn double_index() {
        let mut ds = DoubleStorage::new_with_size(32);
        let mut validator = Vec::with_capacity(32);

        ds.data().iter_mut().enumerate().for_each(|(i, d)| {
            *d = i as f64;
            validator.push(i as f64);
        });

        assert_eq!(31f64, ds[ds.len() - 1]);
    }

    #[test]
    fn float_index_mut() {
        let mut fs = FloatStorage::new_with_size(32);

        for i in 0..fs.len() {
            fs[i] = i as f32;
        }

        assert_eq!(31f32, fs[fs.len() - 1]);
    }

    #[test]
    fn double_index_mut() {
        let mut ds = DoubleStorage::new_with_size(32);

        for i in 0..ds.len() {
            ds[i] = i as f64;
        }

        assert_eq!(31f64, ds[ds.len() - 1]);
    }

    #[test]
    #[ignore]
    fn bench_index() {
        let mut ds = DoubleStorage::new_with_size(1e8 as usize);

        use std::time::Instant;
        let begin = Instant::now();
        let data = ds.data();

        for i in 0..data.len() {
            data[i] = 2f64;
        }

        let end = begin.elapsed();
        println!("Done Mut Slice in {}.{}s", end.as_secs(), end.subsec_millis());
        
        let begin = Instant::now();

        for i in 0..ds.len() {
            ds[i] = 1f64;
        }

        let end = begin.elapsed();
        println!("Done IndexMut in {}.{}s", end.as_secs(), end.subsec_millis());
    }

    #[test]
    fn memory_leak_test() {
        println!("
        This test never fail.
        The only way to check is monitor memory consumption.
        If memory leak, it'll consume about 34GB of memory.
        This shall be large enough to freeze most of PC.
        Otherwise, it'll consume about 1.7GB.
        ");
        // if memory leak, it'll consume about 34GB of memory
        let n = 20;
        for i in 0..n {
            // each new consume about 8 bytes per * size + 1 bool + 1 usize which roughly equals 17 bytes in 64bits system.
            // 63e6 size is estimated to be 1.7GB per each instance
            let mut tmp = DoubleStorage::new_with_size(1e8 as usize);
            // if we didn't fill the value, it may not actually allocate memory
            println!("Mem leak test {}/{}", i, n);
            tmp.fill(1.0);
        }

        println!("Mem leak test done");
    }

    #[test]
    #[ignore]
    fn show_leak() {
        println!("
            This test never fail.
            The only way to check is monitor memory consumption.
            If memory leak, it'll consume about 6.8GB of memory.
            This shall be large enough to freeze most of PC.
            Otherwise, it'll consume about 1.7GB.
        ");
        eprintln!("
            Warning!!! Leaking memory test.
            This test intended leaking memory.
        ");
        // if memory leak, it'll consume about 6.8GB of memory
        let n = 4;
        for i in 0..n {
            // each new consume about 8 bytes per * size + 1 bool + 1 usize which roughly equals 17 bytes in 64bits system.
            // 63e6 size is estimated to be 1.7GB per each instance
            let mut tmp = DoubleStorage::new_with_size(1e8 as usize);
            // if we didn't fill the value, it may not actually allocate memory
            println!("leak test {}/{}", i, n);
            tmp.fill(1.0f64);
            unsafe {
                tmp.forget();
            }
        }

        println!("Mem leak test done");
    }
}