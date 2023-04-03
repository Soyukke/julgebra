#[repr(C)]
#[repr(align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complexf64 {
    pub real: T,
    pub imag: T,
}

impl<T: Copy+One+Default> Complexf64 {
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

impl<T: Default + One> One for Complexf64 {
    fn one() -> Self {
        Self {real: T::one(), imag: T::default()}
    }
}

impl<T: Default> Default for Complexf64 {
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
impl<T: Copy + Neg<Output=T>> Conj for Complexf64 {
    type Output = Complexf64;
    fn conj(&self) -> Complexf64 {
        Complex { real: self.real, imag: -self.imag }

    }
}

impl<T: Add<Output=T>> Add<Complexf64> for Complexf64 {
    type Output = Complexf64;
    fn add(self, rhs: Complexf64) -> Self::Output {
        Complex {real: self.real + rhs.real, imag: self.imag + rhs.imag}
    }
}

impl<T: AddAssign> AddAssign for Complexf64 {
    fn add_assign(&mut self, rhs: Self) {
        self.real += rhs.real;
        self.imag += rhs.imag;
    }
}

// complex<T> * complex<T>
impl<T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T>> Mul<Complexf64> for Complexf64 {
    type Output = Complexf64;
    fn mul(self, rhs: Complexf64) -> Self::Output {
        // (a + bi)(x + yi) = ax + ayi + bxi - by
        Complex {real: self.real*rhs.real - self.imag*rhs.imag, imag: self.real*rhs.imag + self.imag*rhs.real}
    }
}

// complex<T> * T
impl<T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T>> Mul<T> for Complexf64 {
    type Output = Complexf64;
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

impl<T: Default + Copy> Distribution<Complexf64> for Standard
where Standard: Distribution<T>
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complexf64 {
        let (real, imag) = rng.gen::<(T, T)>();
        Complex { real, imag }
    }
}


use std::fmt;
// 出力
impl<T: fmt::Display + Default + std::cmp::PartialOrd + num_traits::Signed> fmt::Display for Complexf64 {
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


