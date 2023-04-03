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


### Example CUDA

```rust
use linear_algebra::{Complex, Matrix, Vector};
use linear_algebra::{CPU,GPU};
let n = 3;
let x = Matrix::<f32>::rand([n, n]);
let y = Matrix::<f32>::rand([n, n]);
let cx = x.gpu();
let cy = y.gpu();
let cz = cx * cy;
let z = cz.unwrap().cpu();
println!("z: {}", z);

=> Result
z: 3x3 Array
 1.03856  0.81080  0.78834 
 1.12424  1.01148  0.45698 
 1.52945  1.29110  0.96300 
```
