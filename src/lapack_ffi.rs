use crate::array::{Array, Vector, Matrix};
use crate::complex::Complex;
use crate::basic_traits::transpose::Transpose;

use lapack_sys::*;
use std::ffi::{c_char, CString};

use std::cmp::min;

pub struct EigenResult<T, S> {
    pub values: Vector<T>,
    pub vectors_l: Matrix<S>,
    pub vectors_r: Matrix<S>,
}

pub trait Eigen {
    type Output;
    fn eig(&mut self) -> Self::Output;
}

impl Matrix<f64> {
    pub fn eig(&mut self) -> EigenResult<Complex<f64>, f64> {
        let jobvl: *mut c_char = CString::new("V").unwrap().into_raw();
        let jobvr: *mut c_char = CString::new("V").unwrap().into_raw();
        let n: i32 = self.dims[0] as i32;
        let u = self.dims[0];
        let lda: i32 = n;
        let ldvl: i32 = n;
        let ldvr: i32 = n;
        let mut wr = Vector::<f64>::zeros([n as usize]);
        let mut wi = Vector::<f64>::zeros([n as usize]);
        let mut vl = Matrix::<f64>::zeros([n as usize, n as usize]);
        let mut vr = Matrix::<f64>::zeros([n as usize, n as usize]);
        let mut work = Vector::<f64>::zeros([1] as [usize; 1]);
        let mut lwork: i32 = -1;
        let mut info: i32 = 0;
        // 実非対称固有値問題
        unsafe {
            // https://docs.rs/lapack-sys/latest/lapack_sys/fn.dgeev_.html
            dgeev_(
                jobvl,
                jobvr,
                &n,
                self.data.as_mut_ptr(),
                &lda,
                wr.data.as_mut_ptr(),
                wi.data.as_mut_ptr(),
                vl.data.as_mut_ptr(),
                &ldvl,
                vr.data.as_mut_ptr(),
                &ldvr,
                work.data.as_mut_ptr(),
                &mut lwork,
                &mut info,
            );

            let mut lwork = work[[0]] as i32;
            let mut work = Vector::<f64>::zeros([lwork as usize]);
            dgeev_(
                jobvl,
                jobvr,
                &n,
                self.data.as_mut_ptr(),
                &lda,
                wr.data.as_mut_ptr(),
                wi.data.as_mut_ptr(),
                vl.data.as_mut_ptr(),
                &ldvl,
                vr.data.as_mut_ptr(),
                &ldvr,
                work.data.as_mut_ptr(),
                &mut lwork,
                &mut info,
            );

        }
        // 固有値
        let mut values: Vector<Complex<f64>> = Vector::zeros([u]);
        for i in 0..u {
            values[[i]] = Complex {real: wr[[i]], imag: wi[[i]]};
        }
        
        EigenResult {values: values, vectors_l: vl, vectors_r: vr}
    }
}


impl Matrix<f64>
where {
    pub fn qr(&self) -> (Matrix<f64>, Matrix<f64>) {
        let q = Matrix::identity([self.dims[0], self.dims[0]]);
        let mut r = self.clone();

        let mut tau = vec![0f64; min(self.dims[0], self.dims[1])];

        let mut info = 0;

        let (n, m) = (self.dims[0] as i32, self.dims[1] as i32);
        unsafe {
            let mut work = vec![0f64; 1];
            // calculate Q from the result of QR decomposition
            let lwork = -1;
            dgeqrf_(
                &n,
                &m,
                r.data.as_mut_ptr(),
                &n,
                tau.as_mut_ptr(),
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
            let mut work = vec![0f64; work[0] as usize];
            dgeqrf_(
                &n,
                &m,
                r.data.as_mut_ptr(),
                &n,
                tau.as_mut_ptr(),
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }

        (q, r)
    }

    pub fn lu(&self) -> (Matrix<f64>, Matrix<f64>, Matrix<f64>) {
        // FIXME: ROW-MajorをColumn-Majorにするためにtransposeしているがメモリ確保して無駄なので直したい
        let mut lu = self.clone().transpose();
        let mut ipvt = Vector::<i32>::zeros([self.dims[0]]);
        let mut info = 0;
        let (n, m) = (self.dims[0] as i32, self.dims[1] as i32);
        unsafe {
            //dgbtrf_(
            dgetrf_(
                &n,
                &m,
                lu.data.as_mut_ptr(),
                &n,
                ipvt.data.as_mut_ptr(),
                &mut info,
            );
        }
        let mut lu = lu.transpose();
        let mut ipvt2 = (ipvt - 1).data;
        let p = Matrix::mutation_matrix(ipvt2);
        let u = Matrix::upper_triangular(&lu);
        let l = Matrix::lower_triangular(&lu);

        (p, l, u)
    }
}
