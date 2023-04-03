use std::{ptr, ffi::{c_float, c_int, c_void, c_char, CString}};

use crate::{Array, Vector, Matrix, Transpose};

use openblas_src::*;
use blas_sys::*;


///
/// 1 10  1  1  =  11 11
/// 1  1  1  1     2   2
pub fn sgemm() {
    unsafe {
        let transa: *mut c_char = CString::new("N").unwrap().into_raw();
        let transb: *mut c_char = CString::new("N").unwrap().into_raw();
                let alpha: c_float = 1.0;
        let beta: c_float = 0.0;
        let mut a = Matrix::ones(2, 2);
        a[(0, 1)] = 10.0;
        let b = Matrix::ones(2, 2);
        let m = a.rows as i32;
        let n = a.cols as i32;
        let k = b.cols as i32;

        let mut c_v = Matrix::zeros(2, 2);

        sgemm_(
            transa,
            transb, 
            &m, 
            &n, 
            &k, 
            &alpha, 
            a.data.as_ptr(), 
            &m, 
            b.data.as_ptr(), 
            &n, 
            &beta, 
            c_v.data.as_mut_ptr(), 
            &k
        );

        println!("sgemm: {}", c_v.transpose());
    }
}
