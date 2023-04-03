use crate::{Matrix, Vector};

pub trait Transpose {
    type Output;
    fn transpose(&self) -> Self::Output;
    fn transpose_mut(self) -> Self::Output;
}

impl<T: Default + Copy> Transpose for Matrix<T> {
    type Output = Matrix<T>;
    fn transpose(&self) -> Self::Output {
        let mut result = Matrix::<T>::zeros([self.dims[1], self.dims[0]]);
        for i in 0..self.dims[0] {
            for j in 0..self.dims[1] {
                result[[j, i]] = self[[i, j]];
            }
        }
        result
    }

    fn transpose_mut(self) -> Self::Output {
        let mut result = Matrix::<T>::zeros([self.dims[1], self.dims[0]]);
        for i in 0..self.dims[0] {
            for j in 0..self.dims[1] {
                result[[j, i]] = self[[i, j]];
            }
        }
        result
    }

}

impl<T: Default + Copy> Transpose for Vector<T> {
    type Output = Matrix<T>;
    fn transpose(&self) -> Self::Output {
        let mut result = Matrix::<T>::zeros([1, self.dims[0]]);
        for i in 0..self.dims[0] {
            result[[0, i]] = self[[i]];
        }
        result
    }

    fn transpose_mut(self) -> Self::Output {
        let mut result = Matrix::<T>::zeros([1, self.dims[0]]);
        for i in 0..self.dims[0] {
            result[[0, i]] = self[[i]];
        }
        result
    }
}
