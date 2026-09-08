//! A tiny 2-4-1 MLP that learns XOR from scratch.
//!
//! One hidden sigmoid layer, mean binary cross-entropy, Adam. The graph is
//! rebuilt every step, keeping each eager tape short-lived like
//! `examples/curve_fit.rs`. Dense layers use rank-2 matmul plus broadcast
//! bias add, both already in the eager graph.

use muninn::autodiff::{Tape, TensorExpr, grad};
use muninn::optim::{Adam, SeedRng, accuracy};
use muninn::span::Span;
use muninn::tensor::Tensor;

const INPUTS: [[f64; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const LABELS: [f64; 4] = [0.0, 1.0, 1.0, 0.0];

fn variable(tape: &Tape, shape: Vec<usize>, data: Vec<f64>) -> TensorExpr {
    tape.variable(Tensor::from_data(shape, data, Span::default()).unwrap())
}

fn dense(input: &TensorExpr, weight: &TensorExpr, bias: &TensorExpr) -> TensorExpr {
    input
        .matmul(weight)
        .unwrap()
        .add(bias)
        .unwrap()
        .sigmoid()
        .unwrap()
}

fn main() {
    let mut rng = SeedRng::new(7);
    let mut init =
        |len: usize| -> Vec<f64> { (0..len).map(|_| rng.next_f64() * 2.0 - 1.0).collect() };
    let mut w1 = init(2 * 4);
    let mut b1 = init(4);
    let mut w2 = init(4);
    let mut b2 = init(1);
    let mut adam = Adam::new(0.1);

    for _ in 0..3000 {
        let tape = Tape::new();
        let w1_var = variable(&tape, vec![2, 4], w1.clone());
        let b1_var = variable(&tape, vec![4], b1.clone());
        let w2_var = variable(&tape, vec![4, 1], w2.clone());
        let b2_var = variable(&tape, vec![1], b2.clone());
        let mut total = tape.scalar(0.0);
        for (point, label) in INPUTS.iter().zip(LABELS.iter()) {
            let input = tape
                .constant(Tensor::from_data(vec![1, 2], point.to_vec(), Span::default()).unwrap());
            let target = tape
                .constant(Tensor::from_data(vec![1, 1], vec![*label], Span::default()).unwrap());
            let hidden = dense(&input, &w1_var, &b1_var);
            let logit = hidden.matmul(&w2_var).unwrap().add(&b2_var).unwrap();
            total = total.add(&logit.bce_with_logits(&target).unwrap()).unwrap();
        }
        let loss = total.mul(&tape.scalar(0.25)).unwrap();
        let grads = [&w1_var, &b1_var, &w2_var, &b2_var]
            .iter()
            .map(|var| grad(&loss, var).unwrap().data().to_vec())
            .collect::<Vec<_>>();
        let mut params = vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()];
        adam.step(&mut params, &grads).unwrap();
        (w1, b1, w2, b2) = (
            params[0].clone(),
            params[1].clone(),
            params[2].clone(),
            params[3].clone(),
        );
    }

    let predicted = INPUTS
        .iter()
        .map(|point| {
            let hidden: Vec<f64> = (0..4)
                .map(|col| {
                    let total: f64 = (0..2).map(|row| point[row] * w1[row * 4 + col]).sum();
                    1.0 / (1.0 + (-(total + b1[col])).exp())
                })
                .collect();
            let logit: f64 = hidden
                .iter()
                .zip(w2.iter())
                .map(|(h, w)| h * w)
                .sum::<f64>()
                + b2[0];
            usize::from(logit > 0.0)
        })
        .collect::<Vec<_>>();
    let expected = LABELS
        .iter()
        .map(|label| *label as usize)
        .collect::<Vec<_>>();
    let acc = accuracy(&predicted, &expected).unwrap();
    println!("xor accuracy after 3000 adam steps: {acc:.3}");
    assert_eq!(acc, 1.0, "mlp should learn xor");
}
