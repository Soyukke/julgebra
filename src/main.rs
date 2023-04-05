fn main() {
    use julgebra::{Complex, Matrix, Vector};
    let n = 100;
    let x = Matrix::<f64>::rand([n, n]);
    let (p, l, u) = x.lu();
    println!("l: {}", l[[10, 10]]);
    println!("u: {}", u[[10, 10]]);
}

