use crate::One;
use crate::Complex;
use std::ops::{Mul, Div, Add, AddAssign, Sub, SubAssign};
use rand::Rng;
use rand::prelude::Distribution;
use rand::distributions::Standard;
use crate::basic_traits::scale::Scale;

pub trait ArrayValue<T>: Default + Clone + One + Copy + Sub<Output = T> + Add<Output = T> + Mul<Output = T> + AddAssign {}
pub trait ArrayValue2<T>: Default + Copy + std::ops::Mul<Output = T> + std::ops::AddAssign<T> {}


#[derive(Debug, Clone)]
pub struct Array<T, const D: usize> {
    pub dims: [usize; D],
    pub data: Vec<T>,
}

pub type Vector<T> = Array<T, 1>;
pub type Matrix<T> = Array<T, 2>;

impl<T:Default+Clone, const D: usize> Array<T, D> {
    pub fn new(dims: [usize; D], init_val: T) -> Self
    {
        let len = dims.into_iter().reduce(|a, b| a*b).unwrap();
        Array {
            dims,
            data: vec![init_val; len],
        }
    }

    pub fn zeros(dims: [usize; D]) -> Self {
        Self::new(dims, T::default())
    }

}

impl<T:Default+Clone+One, const D: usize> Array<T, D>
{
    pub fn ones(dims: [usize; D]) -> Self {
        Self::new(dims, T::one())
    }

    pub fn identity(dims: [usize; D]) -> Self {
        let mut result = Self::zeros(dims);
        let n = *dims.iter().min().unwrap();
        for i in 0..n {
            let idx: [usize; D] = [i; D];
            result[idx] = T::one();
        } 
        result
    }
}

impl<T:Default+Clone, const D: usize> Array<T, D>
where Standard: Distribution<T>
{
    pub fn rand(dims: [usize; D]) -> Self {
        let mut result = Self::new(dims, T::default());
        let mut rng = rand::thread_rng();
        for i in 0..result.data.len() {
                result.data[i] =  rng.gen();
        }
        result
    }
}

impl<T, const D: usize> std::ops::Index<([usize; D])> for Array<T, D> {
    type Output = T;

    fn index(&self, dims: [usize; D]) -> &Self::Output {
        //dims[0] -> reduce(dims[1:end], *)
        //dims[1] -> reduce(dims[2:end], *)
        let n = dims.len();
        let mut idx = 0;
        let mut stride = 1;
        // x[n-1] + x[n-2]*D[n-1] + x[n-3]*D[n-2]*D[n-1] + ... + x[0]*D[1]*D[2]*...*D[n-1]
        for i in (0..n).rev() {
            idx += dims[i] * stride;
            stride *= self.dims[i];
        }
        &self.data[idx]
    }
}

// Row-Major like Z
impl<T , const D: usize> std::ops::IndexMut<[usize; D]> for Array<T, D> {
    fn index_mut(&mut self, dims: [usize; D]) -> &mut Self::Output {
        //println!("data: {:?}", self.data);
        let n = dims.len();
        let mut idx = 0;
        let mut stride = 1;
        // x[n-1] + x[n-2]*D[n-1] + x[n-3]*D[n-2]*D[n-1] + ... + x[0]*D[1]*D[2]*...*D[n-1]
        for i in (0..n).rev() {
            idx += dims[i] * stride;
            stride *= self.dims[i];
        }
        &mut self.data[idx]
    }
}


use std::fmt;

/// # Examples
impl<T: fmt::Display> fmt::Display for Array<T, 1> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = format!("{} Array\n", self.dims[0]);
        let rows = self.dims[0];
        for i in 0..rows {
            s += &format!("{:8.5} ", self[[i]]);
            s += &format!("\n");
        }
        write!(f, "{}", s)
    }
}

/// # Examples
/// 3x3 Array
///     1     1     1 
///     1     1     1 
///     1     1     1 
impl<T: fmt::Display> fmt::Display for Array<T, 2> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = format!("{}x{} Array\n", self.dims[0], self.dims[1]);
        let rows = self.dims[0];
        let cols = self.dims[1];
        for i in 0..rows {
            for j in 0..cols {
                s += &format!("{:8.5} ", self[[i, j]]);
            }
        s += &format!("\n");
        }
        write!(f, "{}", s)
    }
}

/// # Examples
/// 3x3x3 Array
/// [0, :, :]
///     2     2     2 
///     2     2     2 
///     2     2     2 
/// [1, :, :]
///     2     2     2 
///     2     2     2 
///     2     2     2 
/// [2, :, :]
///     2     2     2 
///     2     2     2 
///     2     2     2 
/// 
impl<T: fmt::Display> fmt::Display for Array<T, 3> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let dim0 = self.dims[0];
        let rows = self.dims[0];
        let cols = self.dims[1];
        let mut s = format!("{}x{}x{} Array\n", dim0, rows, cols);
        for a in 0..dim0 {
            s += &format!("[{}, :, :]\n", a);
            for i in 0..rows {
                for j in 0..cols {
                    s += &format!("{:8.5} ", self[[a, i, j]]);
                }
                s += &format!("\n");
            }
        }
        write!(f, "{}", s)
    }
}

//impl<T: Add<Output = T> + AddAssign + Default + Copy> Add<Array<T, 1>> for Array<T, 1> {
//    type Output = Self;
//    fn add(self, other: Self) -> Self::Output {
//        let self_rows = self.dims[0];
//        let other_rows = other.dims[0];
//        if self_rows != other_rows {
//            panic!("matrix size does not match.");
//        }
//        let mut result = Self::zeros([self_rows]);
//        for i in 0..self_rows{
//            result[[i]] = self[[i]] + other[[i]];
//        }
//        result
//    }
//}

impl<T: Default + Clone + Copy + Sub<Output=T>, const D: usize> Sub<T> for Array<T, D> {
    type Output = Self;
    fn sub(self, other: T) -> Self::Output {
        let mut result = Self::zeros(self.dims);
        for i in 0..result.data.len() {
            result.data[i] = self.data[i] - other;
        }
        result
    }
}


/// スカラー乗算
impl<T: Mul<Output = T> + AddAssign + Default + Copy, const D: usize> Mul<T> for Array<T, D> {
    type Output = Array<T, D>;
    fn mul(self, other: T) -> Self::Output {
        self.scale(other)
    }
}


/// スカラー乗算
impl<T: Mul<Output = T> + Div<Output = T> + AddAssign + Default + Copy + One, const D: usize> Div<T> for Array<T, D> {
    type Output = Array<T, D>;
    fn div(self, other: T) -> Self::Output {
        self.scale(T::one()/other)
    }
}

impl<const D: usize> Mul<Array<f32, D>> for f32 {
    type Output = Array<f32, D>;
    fn mul(self, other: Array<f32, D>) -> Self::Output {
        other.scale(self)
    }
}

impl<const D: usize> Mul<Array<f64, D>> for f64 {
    type Output = Array<f64, D>;
    fn mul(self, other: Array<f64, D>) -> Self::Output {
        other.scale(self)
    }
}

impl<const D: usize> Div<Array<f64, D>> for f64 {
    type Output = Array<f64, D>;
    fn div(self, other: Array<f64, D>) -> Self::Output {
        other.scale(1f64/self)
    }
}


impl<const D: usize> Div<Array<f32, D>> for f32 {
    type Output = Array<f32, D>;
    fn div(self, other: Array<f32, D>) -> Self::Output {
        other.scale(1f32/self)
    }
}

impl<const D: usize> Mul<Array<i32, D>> for i32 {
    type Output = Array<i32, D>;
    fn mul(self, other: Array<i32, D>) -> Self::Output {
        other.scale(self)
    }
}

impl<const D: usize> Mul<Array<i64, D>> for i64 {
    type Output = Array<i64, D>;
    fn mul(self, other: Array<i64, D>) -> Self::Output {
        other.scale(self)
    }
}

impl<const D: usize> Mul<Array<Complex<f32>, D>> for Complex<f32> {
    type Output = Array<Complex<f32>, D>;
    fn mul(self, other: Array<Complex<f32>, D>) -> Self::Output {
        other.scale(self)
    }
}

impl<const D: usize> Mul<Array<Complex<f64>, D>> for Complex<f64> {
    type Output = Array<Complex<f64>, D>;
    fn mul(self, other: Array<Complex<f64>, D>) -> Self::Output {
        other.scale(self)
    }
}

impl<T: Mul<Output = T> + AddAssign + Default + Copy> Mul<Matrix<T>> for Vector<T> {
    type Output = Array<T, 2>;
    fn mul(self, other: Matrix<T>) -> Self::Output {
        // (N, 1) * (1, M) = (N, M)
        let n = self.dims[0];
        let (k, m) = (other.dims[0], other.dims[1]);
        if k != 1 {
            panic!("matrix size does not match.");
            //return Err(MatrixError::UndefinedError("matrix size does not match.".to_string()));
        }
        let mut result = Matrix::<T>::zeros([n, m]);
        for i in 0..n {
            for j in 0..m {
                result[[i, j]] = self[[i]] * other[[0, j]];
            }
        }
        result
    }
}

//#[cfg(not(feature="blas"))]
//impl<T: Mul<Output = T> + AddAssign + Default + Copy> Mul<Array<T, 2>> for Array<T, 2> {
//    type Output = Self;
//    fn mul(self, other: Self) -> Self::Output {
//        let (self_rows, self_cols) = (self.dims[0], self.dims[1]);
//        let (other_rows, other_cols) = (other.dims[0], other.dims[1]);
//        if self_cols != other_rows {
//            panic!("matrix size does not match.");
//            //return Err(MatrixError::UndefinedError("matrix size does not match.".to_string()));
//        }
//        let mut result = Self::zeros([self_rows, other_cols]);
//        for i in 0..self_rows{
//            for j in 0..other_cols{
//                let mut x = T::default();
//                for k in 0..other_rows{
//                    x += self[[i, k]] * other[[k, j]]
//                }
//                result[[i, j]] = x;
//            }
//        }
//        result
//    }
//}


/// dot product for vector 
impl<T> Vector<T>
//where T: ArrayValue2<T>
where T: Default + Mul<Output = T> + AddAssign + Copy 
{
    pub fn dot(&self, b: &Vector<T>) -> T {
        let mut z = T::default();
        let n = self.dims[0];
        for i in 0..n {
            z += self[[i]] * b[[i]]
        }
        z
    }
}

impl<T> Matrix<T> 
//where T: ArrayValue2<T>
where T: Default + Mul<Output = T> + AddAssign + Copy 
{
    pub fn dot(&self, b: &Matrix<T>) -> T {
        let mut z = T::default();
        let (n, m) = (self.dims[0], self.dims[1]);
        for i in 0..n {
            for j in 0..m {
                z += self[[i, j]] * b[[i, j]]
            }
        }
        z
    }
}
