use crate::{Matrix, One};
use std::ops::{AddAssign,Mul};

impl<T: Mul<Output = T> + AddAssign + Default + Copy + One> Matrix<T> {

    pub fn switch_row(mut self, pivt1: usize, pivt2: usize) -> Matrix<T> {
        for j in 0..self.dims[1] {
            (self[[pivt1, j]], self[[pivt2, j]]) = (self[[pivt2, j]], self[[pivt1, j]])
            }
        self
    }

    pub fn mutation_matrix(pivt: Vec<i32>) -> Matrix<T> {
        let n = pivt.len();
        let mut r = Matrix::<T>::identity([n, n]);
        for i in 0..n {
            // switch row-i, row-ipvt[i]
            if i < pivt[i] as usize {
                r = r.switch_row(i, pivt[i] as usize);
            }
        }
        r
    }
}
