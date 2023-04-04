## Linear Algebra Rust Implementation

## Usage

See test functions in src/lib.rs.

### Examples

```
=> Rust Code
use crate::matrix::{*};
use crate::One;

let n = Matrix::<f64, 3, 3>::one();
println!("{}", n);

=> Result
3x3 Matrix
    1     0     0 
    0     1     0 
    0     0     1 
```

### Example LAPACK

```rust
let mut A: Array<f64, 2>  = Array::rand([3; 2]);
println!("A: {}", A);
let result = A.eigen();
println!("values: {}", result.values);

=> Result
A: 3x3 Array
 0.21596  0.04488  0.66745 
 0.33858  0.28077  0.37691 
 0.96288  0.61141  0.19091 

values: 3 Array
 1.23647 + 0.00000i 
-0.63434 + 0.00000i 
 0.08552 + 0.00000i 



```

