use julgebla::{Matrix, Transpose};

fn eig() {
    let mut a = Matrix::<f64>::rand([4, 4]);
    let e = a.eig();
    println!("values: {}", e.values);
}


fn lu() {
    let mut a = Matrix::<f64>::rand([4, 4]);
    let (p, l, u) = a.lu();
    println!("a - plu: {}", a - p.transpose()*l*u);
}


fn qr() {
    let mut a = Matrix::<f64>::rand([4, 4]);
    let (q, r) = a.qr();
    println!("a - qr: {}", a - q*r);
}

fn inv() {
    let a = Matrix::<f64>::rand([4, 4]);
    let e = a.inv();
    println!("A^-1: {}", e);
    println!("A*A^-1: {}", &e*&a);
    println!("A*A^-1: {}", &a*&e);
}

fn main() {
    eig();
    inv();
    lu();
    qr();
}
