use std::ops::{Mul, Add, AddAssign};
use crate::{Array, Complex};

pub trait Scale {
    type Input;
    fn scale(self, x: Self::Input) -> Self;
}

impl<T: Mul<Output = T> + AddAssign + Default + Copy, const D: usize> Scale for Array<T, D> {
    type Input = T; 
    fn scale(self, x: Self::Input) -> Self {
        let mut result = Array::<T, D>::zeros(self.dims);
        for i in 0..result.data.len() {
            result.data[i] = x * self.data[i];
        }
        result
    }
}

impl<T: Mul<Output = T> + AddAssign + Default + Copy> Scale for Complex<T> {
    type Input = T; 
    fn scale(self, x: Self::Input) -> Self {
        Complex { real: self.real * x, imag: self.imag * x }
    }
}
