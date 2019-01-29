//! An entry point to create a new tensor.
//! This crate is root crate holding two public sub crates.
//! - storage
//! - tensor
//! 
//! # Simple use case:
//! ## Known size tensor
//! Use `populate_tensor!(type, size, |(index, ref_mut_value)| {});`
//! ### Example
//! To create ByteTensor with 16 elements similar to torch.range(16, dtype=torch.byte):
//! ```Rust
//! let tensor = populate_tensor!(u8, 16, |i, v| {*v = i as u8});
//! ```
//! 
//! To create DoubleTensor with 8 elements similar to torch.range(8, dtype=torch.double):
//! ```Rust
//! let tensor = populate_tensor!(f64, 8, |i, v| {*v = i as f64});
//! ```
//! ## Derive size automatically from closure
//! ```Rust
//! let tensor = populate_tensor!(type, || {});
//! ```
//! where closure return a slice.
//! ### Example
//! To create ByteTensor result of other operation that return a slice of u8.
//! ```Rust
//! let tensor = populate_tensor!(u8, || some_op());
//! ```
//! 
//! To create DoubleTensor with 8 elements similar to torch.range(8, dtype=torch.double):
//! ```Rust
//! let mut buffer: Vec<f64> = Vec::new();
//! 
//! let tensor = populate_tensor!(f64, || {
//!     // load data from file into buffer
//!     // ...
//! 
//!     // finally return a slice
//!     buffer.as_slice()
//! });
//! ```
pub extern crate storage;
pub extern crate tensor;

use tensor::{ByteTensor, CharTensor, DoubleTensor, FloatTensor, IntTensor, LongTensor, ShortTensor, Tensor};

// import generated code from build script.
include!(concat!(env!("OUT_DIR"), "/pop_tensor.rs"));

#[cfg(test)]
mod tests {
    use tensor::*;
    #[test]
    fn test_populate_fixed_char_tensor() {
        
        let tensor = populate_tensor!(i8, 3, |(i, v)| {
            *v = (i * i) as i8;
        });

        assert_eq!(tensor.data(), &[0, 1, 4]);
    }

    #[test]
    fn test_populate_unsized_char_tensor() {
        let data = &[1, 2, 3, 4];
        let tensor = populate_tensor!(i8, || {data});

        assert_eq!(tensor.data(), data);
    }

    #[test]
    fn test_populate_fixed_byte_tensor() {
        
        let tensor = populate_tensor!(u8, 3, |(i, v)| {
            *v = (i * i) as u8;
        });

        assert_eq!(tensor.data(), &[0, 1, 4]);
    }

    #[test]
    fn test_populate_unsized_byte_tensor() {
        let data = &[1, 2, 3, 4];
        let tensor = populate_tensor!(u8, || {data});

        assert_eq!(tensor.data(), data);
    }

    #[test]
    fn test_populate_fixed_float_tensor() {
        
        let tensor = populate_tensor!(f32, 3, |(i, v)| {
            *v = (i * i) as f32;
        });

        assert_eq!(tensor.data(), &[0f32, 1.0, 4.0]);
    }

    #[test]
    fn test_populate_unsized_float_tensor() {
        let data = &[1f32, 2.0, 3.0, 4.0];
        let tensor = populate_tensor!(f32, || {data});

        assert_eq!(tensor.data(), data);
    }

    #[test]
    fn test_populate_fixed_double_tensor() {
        
        let tensor = populate_tensor!(f64, 3, |(i, v)| {
            *v = (i * i) as f64;
        });

        assert_eq!(tensor.data(), &[0f64, 1.0, 4.0]);
    }

    #[test]
    fn test_populate_unsized_double_tensor() {
        let data = &[1f64, 2.0, 3.0, 4.0];
        let tensor = populate_tensor!(f64, || {data});

        assert_eq!(tensor.data(), data);
    }

    #[test]
    fn test_populate_fixed_int_tensor() {
        
        let tensor = populate_tensor!(i32, 3, |(i, v)| {
            *v = (i * i) as i32;
        });

        assert_eq!(tensor.data(), &[0i32, 1, 4]);
    }

    #[test]
    fn test_populate_unsized_int_tensor() {
        let data = &[1i32, 2, 3, 4];
        let tensor = populate_tensor!(i32, || {data});

        assert_eq!(tensor.data(), data);
    }

    #[test]
    fn test_populate_fixed_long_tensor() {
        
        let tensor = populate_tensor!(i64, 3, |(i, v)| {
            *v = (i * i) as i64;
        });

        assert_eq!(tensor.data(), &[0i64, 1, 4]);
    }

    #[test]
    fn test_populate_unsized_long_tensor() {
        let data = &[1i64, 2, 3, 4];
        let tensor = populate_tensor!(i64, || {data});

        assert_eq!(tensor.data(), data);
    }

    #[test]
    fn test_populate_fixed_short_tensor() {
        
        let tensor = populate_tensor!(i16, 3, |(i, v)| {
            *v = (i * i) as i16;
        });

        assert_eq!(tensor.data(), &[0i16, 1, 4]);
    }

    #[test]
    fn test_populate_unsized_short_tensor() {
        // show that data can be owned inside closure.
        // this is because this style of creating copy data, not ref.
        let tensor = populate_tensor!(i16, || {
            let data = &[1i16, 2, 3, 4];
            data
        });

        assert_eq!(tensor.data(), &[1i16, 2, 3, 4]);
    }
}