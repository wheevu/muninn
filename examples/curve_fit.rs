//! A tiny differentiable physics-style curve fit: recover `y = 2x + 1`.
//!
//! The graph is rebuilt for each optimization step, keeping each eager tape
//! short-lived and making the interpreter-only nature of `grad` visible.

use std::error::Error;

use muninn::{Tape, Tensor, grad};

fn main() -> Result<(), Box<dyn Error>> {
    let x = Tensor::from_data(vec![5], vec![0.0, 1.0, 2.0, 3.0, 4.0], Default::default())?;
    let observed = Tensor::from_data(vec![5], vec![1.0, 3.0, 5.0, 7.0, 9.0], Default::default())?;
    let mut slope = 0.0;
    let mut intercept = 0.0;

    for _ in 0..200 {
        let tape = Tape::new();
        let input = tape.constant(x.clone());
        let target = tape.constant(observed.clone());
        let slope_variable = tape.variable(Tensor::scalar(slope));
        let intercept_variable = tape.variable(Tensor::scalar(intercept));
        let prediction = input.mul(&slope_variable)?.add(&intercept_variable)?;
        let residual = prediction.sub(&target)?;
        let loss = residual.mul(&residual)?.sum()?;
        let slope_gradient = grad(&loss, &slope_variable)?.data()[0];
        let intercept_gradient = grad(&loss, &intercept_variable)?.data()[0];

        slope -= 0.01 * slope_gradient;
        intercept -= 0.01 * intercept_gradient;
    }

    println!(
        "fit: y = {:.4}x + {:.4} (target y = 2x + 1)",
        slope, intercept
    );
    Ok(())
}
