use crate::{Matrix, Vector};
use crate::One;
use std::ops::{AddAssign,Mul,Add, Sub, SubAssign, Div};
use rayon::prelude::*;
use itertools::iproduct;

pub struct IntoIterMatrix<T> {
    m: Matrix<T>,
    pos: usize,
}
impl<T: Copy> Iterator for IntoIterMatrix<T> {
    type Item = T; // iterating over an IntoIter should give values moved out of the container (in this case we're copying the same value a few times and pretending they were moved)
    fn next(&mut self) -> Option<Self::Item> {
        if (self.pos < self.m.data.len()) {
            let r = Some(self.m.data[self.pos]);
            self.pos += 1;
            r
        } else {
            None
        }
    }
}

struct IterMutMatrix<'a, T> {
    m: &'a mut Matrix<T>,
    pos: usize 
}

impl<T: Copy> IntoIterator for Matrix<T> {
    type Item = T;
    type IntoIter = IntoIterMatrix<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIterMatrix {m: self, pos: 0}
    }
}

impl Matrix<f64> {
    pub fn lu_r(&self) -> (Matrix<f64>, Matrix<f64>) {
        let (n, m) = (self.dims[0], self.dims[1]);

        let mut l = Matrix::<f64>::identity([n, n]);
        let mut u = self.clone();

        let data: Vec<Vec<(usize, usize, f64, Vec<(usize, f64)>)>> = (0..n).into_par_iter().map(|k| {
            (k+1..n).map(|i| 
                {
                    let coef = u[[i, k]] / u[[k, k]];
                    let us: Vec<(usize, f64)> = (k..n).map(
                        |j| {
                            (j, u[[i, j]] - coef * u[[k, j]])
                        }
                    ).collect();
                    (k, i, coef, us)
                }
            ).collect()
        }).collect();

        data.iter().for_each(|vs| {
            vs.iter().for_each(|(k, i, x, vx)|{
                l[[*i, *k]] = *x;
                vx.iter().for_each(|((j, y))|{
                    u[[*i, *j]] = *y;
                })
            });
        }
        );
       
        (l, u)
    }
}

impl<T: Mul<Output = T> + Sub<Output=T> + Div<Output=T> + AddAssign + Default + Copy + One + SubAssign> Matrix<T> {
    fn det(&self) -> Matrix<T> {
        todo!();
    }

    pub fn mulr(&self, other: &Self) {
        let xs = vec![1, 2, 3, 4, 5];
        let ys: Vec<_> = xs.par_iter().map(|&x| x * x).collect();
        println!("xs: {:?}", xs);
        println!("ys: {:?}", ys);
    }

    
    fn inv(&self) -> Matrix<T> {
        todo!();
    }

    fn svd(&self) -> Matrix<T> {
        todo!();
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut T> {
        self.data.iter_mut()
    }
}
