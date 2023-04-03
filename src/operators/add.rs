use crate::{Matrix,Vector};
use std::ops::{AddAssign,Mul,Add};

impl<T: Add<Output = T> + AddAssign + Default + Copy> Vector<T> {
    fn _add(&self, other: &Self) -> Vector<T> {
        let self_rows = self.dims[0];
        let other_rows = other.dims[0];
        if self_rows != other_rows {
            panic!("size does not match.");
        }
        let mut result = Self::zeros([self_rows]);
        for i in 0..self_rows{
            result[[i]] = self[[i]] + other[[i]];
        }
        result
    }
}

impl<T: Add<Output = T> + AddAssign + Default + Copy> Add<Vector<T>> for Vector<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        self._add(&other)
    }
}

impl<'a, 'b, T: Add<Output = T> + AddAssign + Default + Copy> Add<&'b Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;
    fn add(self, other: &'b Vector<T>) -> Self::Output {
        self._add(other)
    }
}

impl<'a, T: Add<Output = T> + AddAssign + Default + Copy> Add<Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;
    fn add(self, other: Vector<T>) -> Self::Output {
        self._add(&other)
    }
}

impl<'b, T: Add<Output = T> + AddAssign + Default + Copy> Add<&'b Vector<T>> for Vector<T> {
    type Output = Vector<T>;
    fn add(self, other: &'b Vector<T>) -> Self::Output {
        self._add(&other)
    }
}

impl<T: Add<Output = T> + AddAssign + Default + Copy> Matrix<T> {
    fn _add(&self, other: &Self) -> Self {
        let (self_rows, self_cols) = (self.dims[0], self.dims[1]);
        let (other_rows, other_cols) = (other.dims[0], other.dims[1]);
        if self_rows != other_rows || self_cols != other_cols {
            panic!("matrix size does not match.");
        }
        let mut result = Self::zeros([self_rows, other_cols]);
        for i in 0..self_rows{
            for j in 0..other_cols{
                result[[i, j]] = self[[i, j]] + other[[i, j]];
            }
        }
        result
    }
}

impl<T: Add<Output = T> + AddAssign + Default + Copy> Add<Matrix<T>> for Matrix<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        self._add(&other)
    }
}

impl<'a, 'b, T: Add<Output = T> + AddAssign + Default + Copy> Add<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, other: &'b Matrix<T>) -> Self::Output {
        self._add(other)
    }
}

impl<'a, T: Add<Output = T> + AddAssign + Default + Copy> Add<Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, other: Matrix<T>) -> Self::Output {
        self._add(&other)
    }
}

impl<'b, T: Add<Output = T> + AddAssign + Default + Copy> Add<&'b Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, other: &'b Matrix<T>) -> Self::Output {
        self._add(other)
    }
}
