pub trait One {
    fn one() -> Self;
}

impl One for usize {
    fn one() -> Self {
        1
    }
}

impl One for i32 {
    fn one() -> Self{
        1
    }
}

impl One for i64{
    fn one() -> Self {
        1
    }
}


impl One for f32 {
    fn one() -> Self {
        1_f32
    }
}

impl One for f64 {
    fn one() -> Self {
        1_f64
    }
}

