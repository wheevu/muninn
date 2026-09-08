//! Logistic regression on synthetic separable data, twice: once with SGD,
//! once with Adam. Both share the seeded dataset and the same graph
//! recipe: per-sample sigmoid logits with mean binary cross-entropy.

use muninn::autodiff::{Tape, grad};
use muninn::optim::{Adam, SeedRng, Sgd, accuracy, shuffled_indices};
use muninn::span::Span;
use muninn::tensor::Tensor;

fn make_data() -> (Vec<[f64; 2]>, Vec<f64>) {
    let mut rng = SeedRng::new(42);
    let mut inputs = Vec::with_capacity(80);
    let mut labels = Vec::with_capacity(80);
    for _ in 0..80 {
        let x = rng.next_f64() * 4.0 - 2.0;
        let y = rng.next_f64() * 4.0 - 2.0;
        inputs.push([x, y]);
        labels.push(if x + y > 0.0 { 1.0 } else { 0.0 });
    }
    (inputs, labels)
}

fn train<const STEPS: usize>(use_adam: bool) -> (f64, f64) {
    let (inputs, labels) = make_data();
    let mut weights = vec![0.0, 0.0];
    let mut bias = 0.0;
    let mut sgd = Sgd::new(0.5);
    let mut adam = Adam::new(0.05);

    for _ in 0..STEPS {
        let order = shuffled_indices(inputs.len(), 1);
        let tape = Tape::new();
        let weight =
            tape.variable(Tensor::from_data(vec![2], weights.clone(), Span::default()).unwrap());
        let bias_var = tape.variable(Tensor::scalar(bias));
        let mut total = tape.scalar(0.0);
        for index in &order {
            let point = tape.constant(
                Tensor::from_data(vec![2], inputs[*index].to_vec(), Span::default()).unwrap(),
            );
            let label = tape.constant(Tensor::scalar(labels[*index]));
            let logit = point
                .mul(&weight)
                .unwrap()
                .sum()
                .unwrap()
                .add(&bias_var)
                .unwrap();
            total = total.add(&logit.bce_with_logits(&label).unwrap()).unwrap();
        }
        let loss = total.mul(&tape.scalar(1.0 / inputs.len() as f64)).unwrap();
        let weight_grad = grad(&loss, &weight).unwrap();
        let bias_grad = grad(&loss, &bias_var).unwrap();
        let grads = vec![weight_grad.data().to_vec(), bias_grad.data().to_vec()];
        let mut params = vec![weights.clone(), vec![bias]];
        if use_adam {
            adam.step(&mut params, &grads).unwrap();
        } else {
            sgd.step(&mut params, &grads).unwrap();
        }
        weights = params[0].clone();
        bias = params[1][0];
    }

    let predicted = inputs
        .iter()
        .map(|point| usize::from(weights[0] * point[0] + weights[1] * point[1] + bias > 0.0))
        .collect::<Vec<_>>();
    let expected = labels
        .iter()
        .map(|label| *label as usize)
        .collect::<Vec<_>>();
    (accuracy(&predicted, &expected).unwrap(), bias)
}

fn main() {
    let (sgd_acc, _) = train::<200>(false);
    let (adam_acc, _) = train::<200>(true);
    println!("sgd  accuracy after 200 steps:  {sgd_acc:.3}");
    println!("adam accuracy after 200 steps: {adam_acc:.3}");
    assert!(sgd_acc > 0.90, "sgd should separate the data");
    assert!(adam_acc > 0.95, "adam should separate the data");
}
