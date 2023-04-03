
use crate::{Matrix,Vector};

impl<T: Default+Clone+Copy> Vector<T> {
    pub fn to_mat(self) -> Matrix<T> {
        Matrix {dims: [self.dims[0], 1], data: self.data}
    }

    pub fn vcat(&self, other: &Self) -> Vector<T> {
        let self_rows = self.dims[0];
        let other_rows = other.dims[0];
        let mut result = Self::zeros([self_rows+other_rows]);
        for i in 0..self_rows{
            result[[i]] = self[[i]];
        }
        for i in 0..other_rows {
            result[[self_rows+i]] = other[[i]];
        }
        result
    }

    pub fn hcat(&self, other: &Self) -> Matrix<T> {
        let self_rows = self.dims[0];
        let other_rows = other.dims[0];
        if self_rows != other_rows {
            panic!("size does not match.");
        }
        let mut result = Matrix::<T>::zeros([self_rows, 2]);
        for i in 0..self_rows{
            result[[i, 0]] = self[[i]];
            result[[i, 1]] = other[[i]];
        }
        result
    }
}

impl<T: Default+Clone+Copy> Matrix<T> {
    pub fn vcat(&self, other: &Self) -> Matrix<T> {
        let (self_rows, self_cols) = (self.dims[0], self.dims[1]);
        let (other_rows, other_cols) = (other.dims[0], other.dims[1]);
        let other_rows = other.dims[0];
        if self_cols != other_cols {
            panic!("size does not match.");
        }
        let (n, m) = (self_rows+other_rows, self_cols);
        let mut result = Matrix::<T>::zeros([n, m]);
        let result = result.view(0..self_rows, 0..m).set_subarray(self.clone());
        let result = result.view(self_rows..n, 0..m).set_subarray(other.clone());
        result

    }

    pub fn hcat(&self, other: &Self) -> Matrix<T> {
        let (self_rows, self_cols) = (self.dims[0], self.dims[1]);
        let (other_rows, other_cols) = (other.dims[0], other.dims[1]);
        let other_rows = other.dims[0];
        if self_rows != other_rows {
            panic!("size does not match.");
        }
        let (n, m) = (self_rows, self_cols+other_cols);
        let mut result = Matrix::<T>::zeros([n, m]);
        let result = result.view(0..n, 0..self_cols).set_subarray(self.clone());
        let result = result.view(0..n, self_cols..m).set_subarray(other.clone());
        result
    }

}
