//! Library-only first-order training: optimizers, losses, data helpers.
//!
//! Everything here is plain Rust over [`Tensor`] values and the eager
//! [`crate::autodiff`] graph. There is no source-level gradient syntax and
//! no VM or JIT involvement: callers rebuild a [`Tape`] per step, read
//! gradients with [`grad`], and hand flat parameter buffers to an optimizer.
//!
//! Parameters are `Vec<f64>` buffers, one per parameter group. This keeps
//! optimizers free of graph lifetimes: build tensors from buffers,
//! differentiate, then step.

use std::fs;
use std::io;

use crate::autodiff::{AutodiffError, AutodiffErrorKind, Tape, TensorExpr};
use crate::tensor::Tensor;

fn runtime_error(message: impl Into<String>) -> AutodiffError {
    AutodiffError {
        kind: AutodiffErrorKind::Runtime,
        message: message.into(),
    }
}

/// Stochastic gradient descent.
///
/// `momentum` is `0.0` for plain SGD or classically `0.9`. Velocity buffers
/// are allocated lazily on the first step, one per parameter group.
#[derive(Debug, Clone)]
pub struct Sgd {
    lr: f64,
    momentum: f64,
    velocities: Vec<Vec<f64>>,
}

impl Sgd {
    /// Creates plain SGD with the given learning rate.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            momentum: 0.0,
            velocities: Vec::new(),
        }
    }

    /// Sets the momentum coefficient (use `0.0` or typically `0.9`).
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Advances each parameter buffer along its gradient.
    pub fn step(
        &mut self,
        params: &mut [Vec<f64>],
        grads: &[Vec<f64>],
    ) -> Result<(), AutodiffError> {
        check_pair(params, grads, "sgd")?;
        if self.velocities.is_empty() {
            self.velocities = params.iter().map(|group| vec![0.0; group.len()]).collect();
        }
        if self.velocities.len() != params.len() {
            return Err(runtime_error(
                "sgd state does not match the parameter groups",
            ));
        }
        for ((group, gradient), velocity) in params
            .iter_mut()
            .zip(grads.iter())
            .zip(self.velocities.iter_mut())
        {
            if group.len() != gradient.len() || group.len() != velocity.len() {
                return Err(runtime_error(
                    "sgd parameter and gradient lengths must match",
                ));
            }
            for ((parameter, gradient), velocity) in group
                .iter_mut()
                .zip(gradient.iter())
                .zip(velocity.iter_mut())
            {
                if !gradient.is_finite() {
                    return Err(runtime_error("sgd refuses non-finite gradients"));
                }
                *velocity = self.momentum * *velocity + gradient;
                *parameter -= self.lr * *velocity;
            }
        }
        Ok(())
    }
}

/// Shared Adam first/second moment buffers, one per parameter group.
#[derive(Debug, Clone, Default)]
struct AdamState {
    steps: u64,
    moments: Vec<Vec<f64>>,
    velocities: Vec<Vec<f64>>,
    maxima: Vec<Vec<f64>>,
}

impl AdamState {
    fn ensure(&mut self, params: &[Vec<f64>], name: &str) -> Result<(), AutodiffError> {
        if self.moments.is_empty() {
            self.moments = params.iter().map(|group| vec![0.0; group.len()]).collect();
            self.velocities = params.iter().map(|group| vec![0.0; group.len()]).collect();
            self.maxima = params.iter().map(|group| vec![0.0; group.len()]).collect();
        }
        if self.moments.len() != params.len() {
            return Err(runtime_error(format!(
                "{name} state does not match the parameter groups"
            )));
        }
        Ok(())
    }
}

/// Adam with the Kingma and Ba defaults.
///
/// Defaults are `lr = 1e-3`, `betas = (0.9, 0.999)`, `eps = 1e-8`,
/// `weight_decay = 0.0`, `amsgrad = false`. The epsilon follows the original
/// paper and PyTorch rather than Burn's `1e-5`: on small-scale toy losses the
/// smaller epsilon keeps the denominator tight to the second-moment estimate
/// instead of over-dampening updates.
#[derive(Debug, Clone)]
pub struct Adam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    amsgrad: bool,
    state: AdamState,
}

impl Adam {
    /// Creates Adam with `lr` and the documented defaults.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            state: AdamState::default(),
        }
    }

    /// Sets `(beta1, beta2)`.
    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Sets epsilon added to the denominator for numerical stability.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets coupled L2 weight decay, added to the gradient. Prefer
    /// [`AdamW`] for decoupled decay with adaptive methods.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enables the AMSGrad long-term maximum of the second moment.
    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }

    /// Advances each parameter buffer along its Adam update.
    pub fn step(
        &mut self,
        params: &mut [Vec<f64>],
        grads: &[Vec<f64>],
    ) -> Result<(), AutodiffError> {
        check_pair(params, grads, "adam")?;
        self.state.ensure(params, "adam")?;
        self.state.steps += 1;
        let step = self.state.steps as i32;
        let bias1 = 1.0 - self.beta1.powi(step);
        let bias2 = 1.0 - self.beta2.powi(step);
        for (group_index, ((group, gradient), (moment, velocity))) in params
            .iter_mut()
            .zip(grads.iter())
            .zip(
                self.state
                    .moments
                    .iter_mut()
                    .zip(self.state.velocities.iter_mut()),
            )
            .enumerate()
        {
            if group.len() != gradient.len()
                || group.len() != moment.len()
                || group.len() != velocity.len()
            {
                return Err(runtime_error(
                    "adam parameter and gradient lengths must match",
                ));
            }
            for (index, (parameter, gradient)) in group.iter_mut().zip(gradient.iter()).enumerate()
            {
                if !gradient.is_finite() {
                    return Err(runtime_error("adam refuses non-finite gradients"));
                }
                let decayed = gradient + self.weight_decay * *parameter;
                moment[index] = self.beta1 * moment[index] + (1.0 - self.beta1) * decayed;
                velocity[index] =
                    self.beta2 * velocity[index] + (1.0 - self.beta2) * decayed * decayed;
                let corrected_moment = moment[index] / bias1;
                let corrected_velocity = velocity[index] / bias2;
                let denominator = if self.amsgrad {
                    let maximum = &mut self.state.maxima[group_index][index];
                    *maximum = maximum.max(corrected_velocity);
                    maximum.sqrt() + self.eps
                } else {
                    corrected_velocity.sqrt() + self.eps
                };
                *parameter -= self.lr * corrected_moment / denominator;
            }
        }
        Ok(())
    }
}

/// AdamW: Adam with decoupled weight decay.
///
/// L2 regularization and weight decay coincide for plain SGD but not for
/// adaptive methods, so the decay here multiplies the parameter directly
/// instead of entering the gradient moments.
#[derive(Debug, Clone)]
pub struct AdamW {
    inner: Adam,
    decay: f64,
}

impl AdamW {
    /// Creates AdamW with `lr` and decoupled `decay`.
    pub fn new(lr: f64, decay: f64) -> Self {
        Self {
            inner: Adam::new(lr),
            decay,
        }
    }

    /// Sets `(beta1, beta2)`.
    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.inner = self.inner.with_betas(beta1, beta2);
        self
    }

    /// Sets epsilon added to the denominator for numerical stability.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.inner = self.inner.with_eps(eps);
        self
    }

    /// Advances each parameter buffer with decoupled decay applied first.
    pub fn step(
        &mut self,
        params: &mut [Vec<f64>],
        grads: &[Vec<f64>],
    ) -> Result<(), AutodiffError> {
        check_pair(params, grads, "adamw")?;
        let lr = self.inner.lr;
        for (group, gradient) in params.iter_mut().zip(grads.iter()) {
            if group.len() != gradient.len() {
                return Err(runtime_error(
                    "adamw parameter and gradient lengths must match",
                ));
            }
            for (parameter, gradient) in group.iter_mut().zip(gradient.iter()) {
                if !parameter.is_finite() || !gradient.is_finite() {
                    return Err(runtime_error("adamw refuses non-finite values"));
                }
                *parameter -= lr * self.decay * *parameter;
            }
        }
        self.inner.step(params, grads)
    }
}

/// Anything with a tunable learning rate, for schedulers.
pub trait HasLr {
    /// Returns the current learning rate.
    fn lr(&self) -> f64;
    /// Sets the learning rate.
    fn set_lr(&mut self, lr: f64);
}

impl HasLr for Sgd {
    fn lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

impl HasLr for Adam {
    fn lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

impl HasLr for AdamW {
    fn lr(&self) -> f64 {
        self.inner.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.inner.lr = lr;
    }
}

/// Step learning-rate scheduler: every `step_size` steps, `lr *= gamma`.
#[derive(Debug, Clone)]
pub struct StepLr {
    step_size: u64,
    gamma: f64,
    steps: u64,
}

impl StepLr {
    /// Creates a scheduler that decays every `step_size` steps by `gamma`.
    pub fn new(step_size: u64, gamma: f64) -> Self {
        Self {
            step_size: step_size.max(1),
            gamma,
            steps: 0,
        }
    }

    /// Advances one step and updates the optimizer when due. Returns the
    /// learning rate in effect after this step.
    pub fn step<O: HasLr>(&mut self, optimizer: &mut O) -> f64 {
        self.steps += 1;
        if self.steps.is_multiple_of(self.step_size) {
            optimizer.set_lr(optimizer.lr() * self.gamma);
        }
        optimizer.lr()
    }
}

/// Clips every gradient component to `[-bound, bound]`.
pub fn clip_by_value(grads: &mut [Vec<f64>], bound: f64) -> Result<(), AutodiffError> {
    if !bound.is_finite() || bound < 0.0 {
        return Err(runtime_error("clip bound must be finite and non-negative"));
    }
    for group in grads.iter_mut() {
        for gradient in group.iter_mut() {
            *gradient = gradient.clamp(-bound, bound);
        }
    }
    Ok(())
}

/// Clips the global gradient norm to `max_norm`. Returns the norm measured
/// before clipping; gradients are scaled only when it exceeds `max_norm`.
pub fn clip_by_norm(grads: &mut [Vec<f64>], max_norm: f64) -> Result<f64, AutodiffError> {
    if !max_norm.is_finite() || max_norm <= 0.0 {
        return Err(runtime_error("clip norm must be finite and positive"));
    }
    let norm = grads
        .iter()
        .flat_map(|group| group.iter())
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for group in grads.iter_mut() {
            for gradient in group.iter_mut() {
                *gradient *= scale;
            }
        }
    }
    Ok(norm)
}

/// Mean squared error between prediction and target expressions.
///
/// Both sides must share the same shape. Returns a scalar loss with shape
/// `[]`, differentiable in both inputs.
pub fn mse_loss(
    prediction: &TensorExpr,
    target: &TensorExpr,
    tape: &Tape,
) -> Result<TensorExpr, AutodiffError> {
    let residual = prediction.sub(target)?;
    let squares = residual.mul(&residual)?;
    let count = residual.value().data().len().max(1) as f64;
    let total = squares.sum()?;
    total.mul(&tape.scalar(1.0 / count))
}

/// Mean squared error between plain tensors, for reporting.
pub fn mse(prediction: &Tensor, target: &Tensor) -> Result<f64, AutodiffError> {
    if prediction.shape() != target.shape() {
        return Err(AutodiffError {
            kind: AutodiffErrorKind::ShapeMismatch,
            message: format!(
                "mse shape mismatch: {} and {} (prediction and target must share a shape)",
                crate::tensor::format_shape(prediction.shape()),
                crate::tensor::format_shape(target.shape()),
            ),
        });
    }
    let count = prediction.data().len().max(1) as f64;
    let total = prediction
        .data()
        .iter()
        .zip(target.data().iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>();
    Ok(total / count)
}

/// Fraction of predictions matching integer class labels.
pub fn accuracy(predicted: &[usize], labels: &[usize]) -> Result<f64, AutodiffError> {
    if predicted.len() != labels.len() || predicted.is_empty() {
        return Err(runtime_error(
            "accuracy needs non-empty prediction and label slices of equal length",
        ));
    }
    let hits = predicted
        .iter()
        .zip(labels.iter())
        .filter(|(a, b)| a == b)
        .count();
    Ok(hits as f64 / predicted.len() as f64)
}

/// Deterministic 64-bit SplitMix generator for shuffling and splits.
#[derive(Debug, Clone)]
pub struct SeedRng(u64);

impl SeedRng {
    /// Creates a generator from any seed. The same seed replays the same
    /// sequence, which is what makes training harnesses reproducible.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound.max(1) as u64) as usize
    }

    /// Returns a deterministic `f64` in `[0, 1)` for data generation and
    /// weight initialization. Same seed replays the same stream.
    pub fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = (u64::MAX as f64) + 1.0;
        (self.next_u64() as f64) / SCALE
    }
}

/// Returns row indices `0..len` in deterministic shuffled order.
pub fn shuffled_indices(len: usize, seed: u64) -> Vec<usize> {
    let mut rng = SeedRng::new(seed);
    let mut indices: Vec<usize> = (0..len).collect();
    for position in (1..indices.len()).rev() {
        let other = rng.below(position + 1);
        indices.swap(position, other);
    }
    indices
}

/// Splits items into `(train, validation)` with the first
/// `1 - val_fraction` of a seeded shuffle going to train.
pub fn train_val_split<T: Clone>(
    items: &[T],
    val_fraction: f64,
    seed: u64,
) -> Result<(Vec<T>, Vec<T>), AutodiffError> {
    if !(0.0..1.0).contains(&val_fraction) {
        return Err(runtime_error("val_fraction must be in [0, 1)"));
    }
    let order = shuffled_indices(items.len(), seed);
    let val_count = (items.len() as f64 * val_fraction).round() as usize;
    let mut train = Vec::with_capacity(items.len() - val_count);
    let mut validation = Vec::with_capacity(val_count);
    for (rank, index) in order.into_iter().enumerate() {
        if rank < items.len() - val_count {
            train.push(items[index].clone());
        } else {
            validation.push(items[index].clone());
        }
    }
    Ok((train, validation))
}

/// Packs items into consecutive batches of at most `batch_size`.
pub fn batches<T: Clone>(items: &[T], batch_size: usize) -> Result<Vec<Vec<T>>, AutodiffError> {
    if batch_size == 0 {
        return Err(runtime_error("batch_size must be positive"));
    }
    Ok(items
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect())
}

/// Saves parameter groups to a small text format: one group per line as
/// `<len> <v0> <v1> ...` with full `f64` precision.
pub fn save_params(path: &str, params: &[Vec<f64>]) -> io::Result<()> {
    let mut out = String::new();
    out.push_str(&format!("{}\n", params.len()));
    for group in params {
        out.push_str(&format!("{}", group.len()));
        for value in group {
            out.push_str(&format!(" {value:.17e}"));
        }
        out.push('\n');
    }
    fs::write(path, out)
}

/// Loads parameter groups written by [`save_params`].
pub fn load_params(path: &str) -> io::Result<Vec<Vec<f64>>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let groups: usize = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing group count"))?
        .trim()
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad group count"))?;
    let mut params = Vec::with_capacity(groups);
    for _ in 0..groups {
        let line = lines
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing group"))?;
        let mut parts = line.split_whitespace();
        let len: usize = parts
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing group len"))?
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad group len"))?;
        let mut group = Vec::with_capacity(len);
        for _ in 0..len {
            let value: f64 = parts
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing value"))?
                .parse()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad value"))?;
            group.push(value);
        }
        if parts.next().is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "trailing values in group",
            ));
        }
        params.push(group);
    }
    Ok(params)
}

fn check_pair(params: &[Vec<f64>], grads: &[Vec<f64>], name: &str) -> Result<(), AutodiffError> {
    if params.len() != grads.len() {
        return Err(runtime_error(format!(
            "{name} expects one gradient per parameter group"
        )));
    }
    Ok(())
}

/// Builds one linear-regression step on the tape and returns `(loss, grads)`.
#[cfg(test)]
fn linear_step(weights: &[f64], inputs: &[f64], targets: &[f64]) -> (TensorExpr, Vec<f64>, Tape) {
    use crate::autodiff::grad;
    use crate::span::Span;
    let tape = Tape::new();
    let weight = tape.variable(
        Tensor::from_data(vec![weights.len()], weights.to_vec(), Span::default()).unwrap(),
    );
    let input = tape
        .constant(Tensor::from_data(vec![inputs.len()], inputs.to_vec(), Span::default()).unwrap());
    let target = tape.constant(
        Tensor::from_data(vec![targets.len()], targets.to_vec(), Span::default()).unwrap(),
    );
    let prediction = input.mul(&weight).unwrap().sum().unwrap();
    let loss = mse_loss(&prediction, &target, &tape).unwrap();
    let gradient = grad(&loss, &weight).unwrap();
    (loss, gradient.data().to_vec(), tape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_descends_a_quadratic() {
        let mut params = vec![vec![4.0]];
        let mut optimizer = Sgd::new(0.1);
        for _ in 0..50 {
            let grads = vec![vec![2.0 * params[0][0]]];
            optimizer.step(&mut params, &grads).expect("sgd step");
        }
        assert!(params[0][0].abs() < 0.01, "param {}", params[0][0]);
    }

    #[test]
    fn sgd_momentum_still_converges() {
        let mut params = vec![vec![4.0]];
        let mut optimizer = Sgd::new(0.05).with_momentum(0.9);
        for _ in 0..100 {
            let grads = vec![vec![2.0 * params[0][0]]];
            optimizer.step(&mut params, &grads).expect("sgd step");
        }
        assert!(params[0][0].abs() < 0.05, "param {}", params[0][0]);
    }

    #[test]
    fn adam_descends_a_quadratic() {
        let mut params = vec![vec![4.0]];
        let mut optimizer = Adam::new(0.1);
        for _ in 0..200 {
            let grads = vec![vec![2.0 * params[0][0]]];
            optimizer.step(&mut params, &grads).expect("adam step");
        }
        assert!(params[0][0].abs() < 0.05, "param {}", params[0][0]);
    }

    #[test]
    fn adamw_applies_decoupled_decay_without_gradients() {
        let mut params = vec![vec![1.0]];
        let mut optimizer = AdamW::new(0.0, 0.1);
        let grads = vec![vec![0.0]];
        optimizer.step(&mut params, &grads).expect("adamw step");
        assert!(
            (params[0][0] - 1.0).abs() < 1e-12,
            "zero lr means no decay, got {}",
            params[0][0]
        );
        let mut optimizer = AdamW::new(0.1, 0.5);
        optimizer.step(&mut params, &grads).expect("adamw step");
        assert!(
            params[0][0] < 1.0,
            "decay should shrink the weight, got {}",
            params[0][0]
        );
    }

    #[test]
    fn step_lr_decays_on_schedule() {
        let mut optimizer = Sgd::new(0.8);
        let mut scheduler = StepLr::new(2, 0.5);
        assert_eq!(scheduler.step(&mut optimizer), 0.8);
        assert_eq!(scheduler.step(&mut optimizer), 0.4);
        assert_eq!(scheduler.step(&mut optimizer), 0.4);
        assert_eq!(scheduler.step(&mut optimizer), 0.2);
    }

    #[test]
    fn clip_by_norm_bounds_the_norm() {
        let mut grads = vec![vec![3.0, 4.0]];
        let norm = clip_by_norm(&mut grads, 2.5).expect("clip");
        assert_eq!(norm, 5.0);
        let scaled: f64 = grads[0].iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((scaled - 2.5).abs() < 1e-12, "norm {scaled}");
        let mut small = vec![vec![0.6, 0.8]];
        let norm = clip_by_norm(&mut small, 2.5).expect("clip");
        assert_eq!(norm, 1.0);
        assert_eq!(small, vec![vec![0.6, 0.8]]);
    }

    #[test]
    fn clip_by_value_clamps_components() {
        let mut grads = vec![vec![-3.0, 0.5, 2.0]];
        clip_by_value(&mut grads, 1.0).expect("clip");
        assert_eq!(grads, vec![vec![-1.0, 0.5, 1.0]]);
    }

    #[test]
    fn mse_loss_value_and_gradient_agree_with_numbers() {
        let (loss, gradient, _) = linear_step(&[1.0], &[2.0], &[6.0]);
        assert_eq!(loss.value().data(), &[16.0]);
        assert_eq!(gradient, vec![-16.0]);
    }

    #[test]
    fn shuffle_is_deterministic_per_seed() {
        let first = shuffled_indices(16, 7);
        let second = shuffled_indices(16, 7);
        let other = shuffled_indices(16, 8);
        assert_eq!(first, second);
        assert_ne!(first, other);
        let mut sorted = first.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn train_val_split_covers_every_item_once() {
        let items: Vec<usize> = (0..10).collect();
        let (train, validation) = train_val_split(&items, 0.3, 11).expect("split");
        assert_eq!(train.len(), 7);
        assert_eq!(validation.len(), 3);
        let mut seen = train.clone();
        seen.extend_from_slice(&validation);
        seen.sort_unstable();
        assert_eq!(seen, items);
    }

    #[test]
    fn batches_pack_without_dropping_a_tail() {
        let items: Vec<usize> = (0..5).collect();
        let packed = batches(&items, 2).expect("batches");
        assert_eq!(packed, vec![vec![0, 1], vec![2, 3], vec![4]]);
    }

    #[test]
    fn params_save_and_load_roundtrip() {
        let path = std::env::temp_dir().join("muninn-optim-roundtrip.txt");
        let params = vec![vec![1.5, -2.25], vec![0.125]];
        save_params(path.to_str().expect("path"), &params).expect("save");
        let loaded = load_params(path.to_str().expect("path")).expect("load");
        assert_eq!(loaded, params);
        let _ = std::fs::remove_file(path);
    }
}
