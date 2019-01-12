extern crate torcher_derive;

use torcher_derive::TorchStorage;

pub trait TensorStorage<T> : Drop {
    /// Construct new empty Tensor storage
    fn new() -> Self;
    /// Construct new Tensor storage and allocate
    /// fixed size on memory
    fn new_with_size(size: usize) -> Self;

    /// Return a mutable slice of actual memory behind
    /// this storage.
    fn data<'a>(&'a mut self) -> &'a mut [T];

    /// Consume this storage without freeing storage in 
    /// memory. This function is design to be used in FFI case
    /// where the underlying storage will be managed outside 
    /// of Rust.
    /// 
    /// # Safety
    /// Unless you know what you are doing, this will cause
    /// a memory leak. 
    unsafe fn forget(self);

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

// #[TorchStorage(i8)]
// pub struct CharStorage;

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
    fn float_fill_display() {
        let mut fs = FloatStorage::new_with_size(5);
        fs.fill(1.0f32);
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
