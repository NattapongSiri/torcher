#![feature(specialization)]
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

/// A trait that define supported operation on Tensor storage.
/// The actual implementation can be found in build script.
/// The default implementation will always yell `unimplemented!`
pub trait StorageOp{
    type Datatype: Send;

    fn new() -> Storage<Self::Datatype> {
        unimplemented!()
    }
    fn new_with_size(size: usize) -> Storage<Self::Datatype> {
        unimplemented!()
    }
    fn data(&self) -> &[Self::Datatype] {
        unimplemented!()
    }
    fn data_mut(&mut self) -> &mut [Self::Datatype] {
        unimplemented!()
    }
    fn forget(&mut self) {
        unimplemented!()
    }
    fn fill(&mut self, scalar: Self::Datatype) {
        unimplemented!()
    }
    fn resize(&mut self, size: usize) {
        unimplemented!()
    }
    fn retain(&mut self) {
        unimplemented!()
    }
    fn size(&self) -> usize {
        unimplemented!()
    }
    fn swap(&mut self, other: &mut Storage<Self::Datatype>) {
        unimplemented!()
    }
    fn free(&mut self) {
        unimplemented!()
    }
}

/// Default implementation on all other type that is not supported
/// by this `Storage` datatype
impl<T> StorageOp for Storage<T> where T: Send {
    default type Datatype = T;
}

impl<T> Drop for Storage<T> where T: Send {
    fn drop(&mut self) {
        self.free();
    }
}