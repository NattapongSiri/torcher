include!(concat!(env!("OUT_DIR"), "/sys.rs"));

/// Unit struct to represent a ref to Caffe2 storage
struct TorchStorage;

/// A generic Tensor storage.
pub struct Storage<T> where T: Send {
    data: *mut [T],
    forget: bool,
    size: usize,
    storage: *mut TorchStorage
}

pub trait StorageOp<T> where T: Send {
    fn new() -> Storage<T>;
    fn new_with_size(size: usize) -> Storage<T>;
    fn data(&self) -> &[T];
    fn data_mut(&mut self) -> &mut [T];
    fn forget(&mut self);
    fn fill(&mut self, scalar: T);
    fn resize(&mut self, size: usize);
    fn retain(&mut self);
    fn size(&self) -> usize;
    fn swap(&mut self, other: &mut Storage<T>);
    fn free(&mut self);
}

impl<T> Drop for T where for<U> T: StorageOp<U> {
    fn drop(&mut self) {
        self.free();
    }
}