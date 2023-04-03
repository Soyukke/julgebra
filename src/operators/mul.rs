use std::ops::{AddAssign,Mul,Add};
use num_traits::{Float, FromPrimitive};
use openblas_src::*;
use blas_sys::*;
use std::{ptr, ffi::{c_float, c_int, c_void, c_char, CString}};
use crate::array::{Matrix, Vector};

trait BlasOps {
    fn _mul(&self, other: &Self) -> Self;
}
impl BlasOps for Matrix<f32> {
    fn _mul(&self, other: &Self) -> Matrix<f32> {
        // A^T * B^T = C^T
        let transa: *mut c_char = CString::new("N").unwrap().into_raw();
        let transb: *mut c_char = CString::new("N").unwrap().into_raw();
        let alpha:c_float= 1.0;
        let beta: c_float = 0.0;

        let m = self.dims[0] as i32;
        let n = self.dims[1] as i32;
        let k = other.dims[1] as i32;

        let mut result: Matrix<f32> = Matrix::zeros([m as usize, n as usize]);

        unsafe {
            // https://netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html
            let status = sgemm_(
                transa,
                transb, 
                &m, 
                &n, 
                &k, 
                &alpha, 
                self.data.as_ptr(), 
                &m, 
                other.data.as_ptr(), 
                &n, 
                &beta, 
                result.data.as_mut_ptr(), 
                &k
            );
        }
        result
    }
}


 impl BlasOps for Matrix<f64> {
   
    fn _mul(&self, other: &Self) -> Matrix<f64> {
        // A^T * B^T = C^T
        let transa: *mut c_char = CString::new("N").unwrap().into_raw();
        let transb: *mut c_char = CString::new("N").unwrap().into_raw();
        let alpha:f64= 1.0;
        let beta: f64= 0.0;

        let m = self.dims[0] as i32;
        let n = self.dims[1] as i32;
        let k = other.dims[1] as i32;

        let mut result: Matrix<f64> = Matrix::zeros([m as usize, n as usize]);

        unsafe {
            // https://netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html
            let status = dgemm_(
                transa,
                transb, 
                &m, 
                &n, 
                &k, 
                &alpha, 
                self.data.as_ptr(), 
                &m, 
                other.data.as_ptr(), 
                &n, 
                &beta, 
                result.data.as_mut_ptr(), 
                &k
            );
        }
        result
    }
}

impl<'a, 'b, T> Mul<&'b Matrix<T>> for &'a Matrix<T>
where Matrix<T>: BlasOps
{
    type Output = Matrix<T>;
    fn mul(self, other: &'b Matrix<T>) -> Self::Output {
        self._mul(other)
    }
}

impl<'a, T> Mul<Matrix<T>> for &'a Matrix<T>
where Matrix<T>: BlasOps
{
    type Output = Matrix<T>;
    fn mul(self, other: Matrix<T>) -> Self::Output {
        self._mul(&other)
    }
}


impl<'b, T> Mul<&'b Matrix<T>> for Matrix<T>
where Matrix<T>: BlasOps
{
    type Output = Matrix<T>;
    fn mul(self, other: &'b Matrix<T>) -> Self::Output {
        self._mul(other)
    }
}

impl<T> Mul<Matrix<T>> for Matrix<T>
where Matrix<T>: BlasOps
{
    type Output = Matrix<T>;
    fn mul(self, other: Matrix<T>) -> Self::Output {
        self._mul(&other)
    }
}

//impl<'a, 'b> Mul<&'b Matrix<f32>> for &'a Matrix<f32>
//{
//    type Output = Matrix<f32>;
//    fn mul(self, other: &'b Matrix<f32>) -> Self::Output {
//        self._mul(other)
//    }
//}

//impl<'a, 'b> Mul<&'b Matrix<f64>> for &'a Matrix<f64>
//{
//    type Output = Matrix<f64>;
//    fn mul(self, other: &'b Matrix<f64>) -> Self::Output {
//        self._mul(other)
//    }
//}



//impl<'b, T: JFloat> for Matrix<T> {
//    type Output = Matrix<T>;
//    fn mul(self, other: &'b Matrix<T>) -> Self::Output {
//        self._mul(other)
//    }
//}
//
//impl<'a, T: JFloat> for &'a Matrix<T> {
//    type Output = Matrix<T>;
//    fn mul(self, other: Matrix<T>) -> Self::Output {
//        self._mul(&other)
//    }
//}


//
//impl<T: Mul<Output = T> + AddAssign + Default + Copy> Mul<Vector<T>> for Matrix<T> {
//    type Output = Vector<T>;
//    fn mul(self, other: Vector<T>) -> Self::Output {
//        self._mul_vec(&other)
//    }
//}
//
//impl<'a, 'b, T: Mul<Output = T> + AddAssign + Default + Copy> Mul<&'b Vector<T>> for &'a Matrix<T> {
//    type Output = Vector<T>;
//    fn mul(self, other: &'b Vector<T>) -> Self::Output {
//        self._mul_vec(other)
//    }
//}
//
//impl<'a, T: Mul<Output = T> + AddAssign + Default + Copy> Mul<Vector<T>> for &'a Matrix<T> {
//    type Output = Vector<T>;
//    fn mul(self, other: Vector<T>) -> Self::Output {
//        self._mul_vec(&other)
//    }
//}
//
//impl<'b, T: Mul<Output = T> + AddAssign + Default + Copy> Mul<&'b Vector<T>> for Matrix<T> {
//    type Output = Vector<T>;
//    fn mul(self, other: &'b Vector<T>) -> Self::Output {
//        self._mul_vec(other)
//    }
//}
