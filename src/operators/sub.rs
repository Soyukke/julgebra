use crate::{Matrix,Vector};
use std::ops::{Mul, Add, AddAssign, Sub, SubAssign};

impl<T: Sub<Output = T> + SubAssign + Default + Copy> Vector<T> {
    fn _sub(&self, other: &Self) -> Self {
        let self_rows = self.dims[0];
        let other_rows = other.dims[0];
        if self_rows != other_rows {
            panic!("matrix size does not match.");
        }
        let mut result = Self::zeros([self_rows]);
        for i in 0..self_rows{
            result[[i]] = self[[i]] - other[[i]];
        }
        result
    }
}

impl<T: Sub<Output = T> + SubAssign + Default + Copy> Sub<Vector<T>> for Vector<T> {
    type Output = Vector<T>;
    fn sub(self, other: Self) -> Self::Output {
        self._sub(&other)
    }
}

impl<'a, 'b, T: Sub<Output = T> + SubAssign + Default + Copy> Sub<&'b Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;
    fn sub(self, other: &'b Vector<T>) -> Self::Output {
        self._sub(other)
    }
}

impl<'a, T: Sub<Output = T> + SubAssign + Default + Copy> Sub<Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;
    fn sub(self, other: Vector<T>) -> Self::Output {
        self._sub(&other)
    }
}


impl<'b, T: Sub<Output = T> + SubAssign + Default + Copy> Sub<&'b Vector<T>> for Vector<T> {
    type Output = Vector<T>;
    fn sub(self, other: &'b Vector<T>) -> Self::Output {
        self._sub(other)
    }
}


impl<T: Sub<Output = T> + SubAssign + Default + Copy> Matrix<T> {
    fn _sub(&self, other: &Self) -> Self {
        let (self_rows, self_cols) = (self.dims[0], self.dims[1]);
        let (other_rows, other_cols) = (other.dims[0], other.dims[1]);
        if self_rows != other_rows || self_cols != other_cols {
            panic!("matrix size does not match.");
        }
        let mut result = Self::zeros([self_rows, other_cols]);
        for i in 0..self_rows{
            for j in 0..other_cols{
                result[[i, j]] = self[[i, j]] - other[[i, j]];
            }
        }
        result
    }
}

impl<T: Sub<Output = T> + SubAssign + Default + Copy> Sub<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: Self) -> Self::Output {
        self._sub(&other)
    }
}

impl<'a, 'b, T: Sub<Output = T> + SubAssign + Default + Copy> Sub<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: &'b Matrix<T>) -> Self::Output {
        self._sub(other)
    }
}


impl<'a, T: Sub<Output = T> + SubAssign + Default + Copy> Sub<Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: Matrix<T>) -> Self::Output {
        self._sub(&other)
    }
}

impl<'b, T: Sub<Output = T> + SubAssign + Default + Copy> Sub<&'b Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: &'b Matrix<T>) -> Self::Output {
        self._sub(other)
    }
}
