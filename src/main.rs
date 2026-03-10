use std::{env, fs};

use muninn::compile_and_run;

const DEMO_PROGRAM: &str = r#"
fn checked_scale(scale: Float) -> Option[Float] {
    if (scale == 0.0) { none } else { some(scale) }
}

fn weighted_signal(raw: Float[3], weights: Float[3], norm: Float) -> Float[3] {
    let normalized: Float[3] = raw / norm;
    let weighted: Float[3] = normalized * weights;
    0.95 * (weighted + 0.05)
}

fn forward(raw: Float[3], weights: Float[3], bias: Float, norm: Float) -> Option[Float] {
    let safe_norm: Float = checked_scale(norm)?;
    let feats: Float[3] = weighted_signal(raw, weights, safe_norm);
    let score: Float = feats[0] + feats[1] + feats[2] + bias;
    some(unless (score > 0.0) { 0.0 } else { 1.0 })
}

let output = forward([210.0, 140.0, 70.0], [0.2, -0.5, 0.1], 0.3, 255.0);
print("demo output = {output}");
"#;

fn main() {
    let source = match env::args().nth(1) {
        Some(path) => match fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(err) => {
                eprintln!("failed to read '{path}': {err}");
                std::process::exit(1);
            }
        },
        None => DEMO_PROGRAM.to_string(),
    };

    match compile_and_run(&source) {
        Ok(value) => {
            println!("=> {}", value);
        }
        Err(errors) => {
            for error in errors {
                eprintln!("{}", error.render_with_source(&source));
            }
            std::process::exit(1);
        }
    }
}
