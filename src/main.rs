fn main() {
    use julgebla::{Complex, Matrix, Vector};
    let n = 100;
    let x = Matrix::<f64>::rand([n, n]);
    let (l, u) = x.lu();
    println!("l: {}", l[[10, 10]]);
    println!("u: {}", u[[10, 10]]);
}

