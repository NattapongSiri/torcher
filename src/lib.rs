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

impl<'a> ToTensor<FloatTensor<'a>, f32> for [f32]
{
    fn tensor(&self) -> FloatTensor<'a> {
        let n = self.len();
        let mut tensor = FloatTensor::new_with_size_1d(n);
        let data = tensor.data_mut();
        self.iter().enumerate().for_each(|(i, f)| data[i] = *f);

        tensor
    }
}

impl<'a> ToTensor<DoubleTensor<'a>, f64> for [f64] {
    fn tensor(&self) -> DoubleTensor<'a> {
        let n = self.len();
        let mut tensor = DoubleTensor::new_with_size_1d(n);
        let data = tensor.data_mut();
        self.iter().enumerate().for_each(|(i, d)| data[i] = *d);

        tensor
    }
}

impl<'a> ToTensor<ByteTensor<'a>, u8> for [u8] {
    fn tensor(&self) -> ByteTensor<'a> {
        let n = self.len();
        let mut tensor = ByteTensor::new_with_size_1d(n);
        let data = tensor.data_mut();
        self.iter().enumerate().for_each(|(i, b)| data[i] = *b);

        tensor
    }
}

impl<'a> ToTensor<CharTensor<'a>, i8> for [i8] {
    fn tensor(&self) -> CharTensor<'a> {
        let n = self.len();
        let mut tensor = CharTensor::new_with_size_1d(n);
        let data = tensor.data_mut();
        self.iter().enumerate().for_each(|(i, c)| data[i] = *c);

        tensor
    }
}

impl<'a> ToTensor<IntTensor<'a>, i32> for [i32] {
    fn tensor(&self) -> IntTensor<'a> {
        let n = self.len();
        let mut tensor = IntTensor::new_with_size_1d(n);
        let data = tensor.data_mut();
        self.iter().enumerate().for_each(|(i, v)| data[i] = *v);

        tensor
    }
}

impl<'a> ToTensor<LongTensor<'a>, i64> for [i64] {
    fn tensor(&self) -> LongTensor<'a> {
        let n = self.len();
        let mut tensor = LongTensor::new_with_size_1d(n);
        let data = tensor.data_mut();
        self.iter().enumerate().for_each(|(i, v)| data[i] = *v);

        tensor
    }
}

impl<'a> ToTensor<ShortTensor<'a>, i16> for [i16] {
    fn tensor(&self) -> ShortTensor<'a> {
        let n = self.len();
        let mut tensor = ShortTensor::new_with_size_1d(n);
        let data = tensor.data_mut();
        self.iter().enumerate().for_each(|(i, v)| data[i] = *v);

        tensor
    }
}