// WASM-portable when built without the "std" feature.
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Compute RMS LayerNorm in-place.
///
/// Normalises each row of `hidden_size` elements by its root-mean-square, then
/// scales by learned `weight`. No mean subtraction (unlike LayerNorm).
///
/// Formula: `y[i] = (x[i] / rms) * weight[i]`  where `rms = sqrt(mean(x²) + eps)`
///
/// # Example
/// ```
/// use blitz_rmsnorm::rms_norm;
/// let mut x = vec![3.0_f32, 4.0];
/// let weight = vec![1.0_f32; 2];
/// rms_norm(&mut x, &weight, 2, 1e-6);
/// let rms = (12.5_f32).sqrt();
/// assert!((x[0] - 3.0/rms).abs() < 1e-4);
/// ```
pub fn rms_norm(x: &mut [f32], weight: &[f32], hidden_size: usize, eps: f32) {
    assert_eq!(weight.len(), hidden_size, "weight length must equal hidden_size");
    assert_eq!(x.len() % hidden_size, 0, "x length must be divisible by hidden_size");

    let rows = x.len() / hidden_size;
    for r in 0..rows {
        let row = &mut x[r * hidden_size..(r + 1) * hidden_size];
        let mean_sq: f32 = row.iter().map(|&v| v * v).sum::<f32>() / hidden_size as f32;
        let rms_inv = 1.0 / (mean_sq + eps).sqrt();
        for (v, &w) in row.iter_mut().zip(weight.iter()) {
            *v = (*v * rms_inv) * w;
        }
    }
}

/// Non-mutating variant — returns a new normalised buffer.
///
/// # Example
/// ```
/// use blitz_rmsnorm::rms_norm_into;
/// let x = vec![1.0_f32; 4];
/// let weight = vec![1.0_f32; 4];
/// let out = rms_norm_into(&x, &weight, 4, 1e-6);
/// assert_eq!(out.len(), 4);
/// ```
pub fn rms_norm_into(x: &[f32], weight: &[f32], hidden_size: usize, eps: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    rms_norm(&mut out, weight, hidden_size, eps);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() < tol }

    #[test]
    fn unit_weight_output_rms_is_one() {
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];
        rms_norm(&mut x, &[1.0; 4], 4, 1e-8);
        let rms: f32 = (x.iter().map(|&v| v*v).sum::<f32>() / 4.0).sqrt();
        assert!(approx(rms, 1.0, 1e-5), "rms={rms}");
    }

    #[test]
    fn known_two_element_row() {
        // RMS([3,4]) = sqrt(12.5)
        let mut x = vec![3.0_f32, 4.0];
        rms_norm(&mut x, &[1.0, 1.0], 2, 1e-8);
        let r = 12.5_f32.sqrt();
        assert!(approx(x[0], 3.0/r, 1e-5));
        assert!(approx(x[1], 4.0/r, 1e-5));
    }

    #[test]
    fn weight_doubles_output() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let o1 = rms_norm_into(&x, &[1.0; 4], 4, 1e-8);
        let o2 = rms_norm_into(&x, &[2.0; 4], 4, 1e-8);
        for i in 0..4 { assert!(approx(o2[i], o1[i]*2.0, 1e-5), "i={i}"); }
    }

    #[test]
    fn uniform_input_output_equals_weight() {
        let mut x = vec![5.0_f32; 8];
        let w: Vec<f32> = (1..=8).map(|i| i as f32 * 0.5).collect();
        rms_norm(&mut x, &w, 8, 1e-8);
        for i in 0..8 { assert!(approx(x[i], w[i], 1e-5), "i={i}"); }
    }

    #[test]
    fn zero_input_stays_finite() {
        let mut x = vec![0.0_f32; 4];
        rms_norm(&mut x, &[1.0; 4], 4, 1e-6);
        for &v in &x { assert!(v.is_finite(), "non-finite: {v}"); }
    }

    #[test]
    fn two_identical_rows_normalise_identically() {
        let mut x: Vec<f32> = [1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0].to_vec();
        rms_norm(&mut x, &[1.0; 4], 4, 1e-8);
        for i in 0..4 { assert!(approx(x[i], x[i+4], 1e-6), "i={i}"); }
    }

    #[test]
    fn length_preserved() {
        let out = rms_norm_into(&[0.5f32; 16], &[1.0; 4], 4, 1e-6);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn inplace_matches_functional() {
        let x: Vec<f32> = (1..=8).map(|i| i as f32 * 0.3).collect();
        let w = vec![1.0_f32; 8];
        let f = rms_norm_into(&x, &w, 8, 1e-6);
        let mut ip = x.clone();
        rms_norm(&mut ip, &w, 8, 1e-6);
        assert_eq!(f, ip);
    }

    #[test]
    fn single_element_row() {
        let mut x = vec![4.0_f32];
        rms_norm(&mut x, &[3.0], 1, 1e-8);
        assert!(approx(x[0], 3.0, 1e-5), "got {}", x[0]);
    }

    #[test]
    fn negative_values_sign_preserved() {
        let mut x = vec![-1.0_f32, -2.0, -3.0, -4.0];
        rms_norm(&mut x, &[1.0; 4], 4, 1e-8);
        for &v in &x { assert!(v < 0.0, "sign flipped: {v}"); }
        let rms: f32 = (x.iter().map(|&v| v*v).sum::<f32>() / 4.0).sqrt();
        assert!(approx(rms, 1.0, 1e-5));
    }

    #[test]
    fn large_hidden_size_4096() {
        let h = 4096;
        let mut x: Vec<f32> = (0..h).map(|i| i as f32 * 0.001).collect();
        rms_norm(&mut x, &vec![1.0f32; h], h, 1e-6);
        let rms: f32 = (x.iter().map(|&v| v*v).sum::<f32>() / h as f32).sqrt();
        assert!(approx(rms, 1.0, 1e-3), "rms={rms}");
    }

    #[test]
    #[should_panic]
    fn panics_weight_mismatch() {
        let mut x = vec![1.0f32; 4];
        rms_norm(&mut x, &[1.0; 3], 4, 1e-6);
    }

    #[test]
    #[should_panic]
    fn panics_indivisible_length() {
        let mut x = vec![1.0f32; 5];
        rms_norm(&mut x, &[1.0; 4], 4, 1e-6);
    }
}
