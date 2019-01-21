pub extern crate storage;
pub extern crate tensor;

use tensor::{ByteTensor, CharTensor, DoubleTensor, FloatTensor, IntTensor, LongTensor, ShortTensor, Tensor};

/// Convert existing type into Tensor derivative type.
/// This is a copy operation which will copy every
/// data from self to return Tensor.
pub trait ToTensor<T, U> 
where T: Tensor<U>
{
    fn tensor(&self) -> T;
}

impl ToTensor<FloatTensor, f32> for [f32]
{
    fn tensor(&self) -> FloatTensor {
        let n = self.len();
        let mut tensor = FloatTensor::new_with_size_1d(n);
        let data = tensor.data();
        self.iter().enumerate().for_each(|(i, f)| data[i] = *f);

        tensor
    }
}

impl ToTensor<DoubleTensor, f64> for [f64] {
    fn tensor(&self) -> DoubleTensor {
        let n = self.len();
        let mut tensor = DoubleTensor::new_with_size_1d(n);
        let data = tensor.data();
        self.iter().enumerate().for_each(|(i, d)| data[i] = *d);

        tensor
    }
}

impl ToTensor<ByteTensor, u8> for [u8] {
    fn tensor(&self) -> ByteTensor {
        let n = self.len();
        let mut tensor = ByteTensor::new_with_size_1d(n);
        let data = tensor.data();
        self.iter().enumerate().for_each(|(i, b)| data[i] = *b);

        tensor
    }
}

impl ToTensor<CharTensor, i8> for [i8] {
    fn tensor(&self) -> CharTensor {
        let n = self.len();
        let mut tensor = CharTensor::new_with_size_1d(n);
        let data = tensor.data();
        self.iter().enumerate().for_each(|(i, c)| data[i] = *c);

        tensor
    }
}

impl ToTensor<IntTensor, i32> for [i32] {
    fn tensor(&self) -> IntTensor {
        let n = self.len();
        let mut tensor = IntTensor::new_with_size_1d(n);
        let data = tensor.data();
        self.iter().enumerate().for_each(|(i, v)| data[i] = *v);

        tensor
    }
}

impl ToTensor<LongTensor, i64> for [i64] {
    fn tensor(&self) -> LongTensor {
        let n = self.len();
        let mut tensor = LongTensor::new_with_size_1d(n);
        let data = tensor.data();
        self.iter().enumerate().for_each(|(i, v)| data[i] = *v);

        tensor
    }
}

impl ToTensor<ShortTensor, i16> for [i16] {
    fn tensor(&self) -> ShortTensor {
        let n = self.len();
        let mut tensor = ShortTensor::new_with_size_1d(n);
        let data = tensor.data();
        self.iter().enumerate().for_each(|(i, v)| data[i] = *v);

        tensor
    }
}