use std::ops::{Add, Sub, Mul, AddAssign, Neg};
use crate::One;
use rand::Rng;
use rand::distributions::{Distribution, Standard};

/**
 * Complex<T>を定義する。
 * Add, Mul, AddAssign traitなどを実装することで、Matrix演算のトレイト境界を満たすことができ、
 * 複素行列の演算も自然に実装できる。
 */

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex<T> {
    pub real: T,
    pub imag: T,
}

pub type c64 = Complex<f64>;
pub type c32 = Complex<f32>;


impl<T: Copy+One+Default> Complex<T> {
    pub fn new(real:T, imag:T) -> Self {
        Self { real: real, imag: imag  }
    }

    // 虚数
    pub fn i() -> Self {
        Self {real: T::default(), imag: T::one()}
    }

    pub fn to_vec(&self) -> Vec<T> {
        vec![self.real, self.imag]
    }
}

impl<T: Default + One> One for Complex<T> {
    fn one() -> Self {
        Self {real: T::one(), imag: T::default()}
    }
}

impl<T: Default> Default for Complex<T> {
    fn default() -> Self {
        Self {real: T::default(), imag: T::default()}
    }
}

/// # 複素共役
pub trait Conj{
    type Output;
    fn conj(&self) -> Self::Output;
}

// 複素共役の実装。Tのトレイト境界=制約はCopyとNeg(マイナスの単項演算子)
impl<T: Copy + Neg<Output=T>> Conj for Complex<T> {
    type Output = Complex<T>;
    fn conj(&self) -> Complex<T> {
        Complex { real: self.real, imag: -self.imag }

    }
}

impl<T: Add<Output=T>> Add<Complex<T>> for Complex<T> {
    type Output = Complex<T>;
    fn add(self, rhs: Complex<T>) -> Self::Output {
        Complex {real: self.real + rhs.real, imag: self.imag + rhs.imag}
    }
}

impl<T: AddAssign> AddAssign for Complex<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.real += rhs.real;
        self.imag += rhs.imag;
    }
}

// complex<T> * complex<T>
impl<T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T>> Mul<Complex<T>> for Complex<T> {
    type Output = Complex<T>;
    fn mul(self, rhs: Complex<T>) -> Self::Output {
        // (a + bi)(x + yi) = ax + ayi + bxi - by
        Complex {real: self.real*rhs.real - self.imag*rhs.imag, imag: self.real*rhs.imag + self.imag*rhs.real}
    }
}

// complex<T> * T
impl<T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T>> Mul<T> for Complex<T> {
    type Output = Complex<T>;
    fn mul(self, rhs: T) -> Self::Output {
        // (a + bi)(x + yi) = ax + ayi + bxi - by
        Complex {real: self.real*rhs, imag: self.imag*rhs}
    }
}

impl Mul<Complex<i32>> for i32
{
    type Output = Complex<i32>;
    fn mul(self, rhs: Complex<i32>) -> Self::Output {
        // (a + bi)(x + yi) = ax + ayi + bxi - by
        Complex {real: rhs.real*self, imag: rhs.imag*self}
    }
}

impl Mul<Complex<f32>> for f32
{
    type Output = Complex<f32>;
    fn mul(self, rhs: Complex<f32>) -> Self::Output {
        Complex {real: rhs.real*self, imag: rhs.imag*self}
    }
}

impl Mul<Complex<f64>> for f64
{
    type Output = Complex<f64>;
    fn mul(self, rhs: Complex<f64>) -> Self::Output {
        Complex {real: rhs.real*self, imag: rhs.imag*self}
    }
}

impl<T: Default + Copy> Distribution<Complex<T>> for Standard
where Standard: Distribution<T>
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex<T> {
        let (real, imag) = rng.gen::<(T, T)>();
        Complex { real, imag }
    }
}


use std::fmt;
// 出力
impl<T: fmt::Display + Default + std::cmp::PartialOrd + num_traits::Signed> fmt::Display for Complex<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let sign = if self.imag < T::default() {
            "-"
        } else {
            "+"
        };
        let s = format!("{:8.5} {} {:7.5}i", self.real, sign, self.imag.abs());
        write!(f, "{}", s)
    }
}


