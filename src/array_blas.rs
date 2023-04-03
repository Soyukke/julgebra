use std::ops::{Mul, AddAssign};
use openblas_src::*;
use blas_sys::*;
use std::{ptr, ffi::{c_float, c_int, c_void, c_char, CString}};
use crate::array::{Array, Matrix};




impl Mul<Array<f64, 2>> for Array<f64, 2> {

    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        // A^T * B^T = C^T
        let transa: *mut c_char = CString::new("N").unwrap().into_raw();
        let transb: *mut c_char = CString::new("N").unwrap().into_raw();
        let alpha:f64= 1.0;
        let beta: f64= 0.0;

        let m = self.dims[0] as i32;
        let n = self.dims[1] as i32;
        let k = other.dims[1] as i32;

        let mut result: Array<f64, 2> = Array::zeros([m as usize, n as usize]);

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


