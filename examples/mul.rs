use julgebra::*;

fn mul_c() {
    let n = 3;
    let a = Matrix::<c32>::rand([n, n]);
    let b = Matrix::<c32>::rand([n, n]);
    let c = a * b;
    println!("c: {}", c);
}

fn mul_c2() {
    let n = 3;
    let a = Matrix::<c64>::rand([n, n]);
    let b = Matrix::<c64>::rand([n, n]);
    let c = a * b;
    println!("c: {}", c);
}

fn mul() {
    let n = 2;
    let mut a = Matrix::<f32>::zeros([n, n]);
    let mut b = Matrix::<f32>::zeros([n, n]);
    a[[0, 1]] = 1f32;
    a[[1, 0]] = 2f32;
    a[[1, 1]] = 1f32;

    b[[0, 1]] = 1f32;
    b[[1, 0]] = 1f32;
    b[[1, 1]] = 1f32;

    let c = &a * &b;
    println!("a: {}", a);
    println!("b: {}", b);
    println!("c: {}", c);
}

fn main() {
    mul_c();
    mul_c2();
    mul();
}
