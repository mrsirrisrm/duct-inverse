//! Forward-mode automatic differentiation via dual numbers.
//!
//! A `Dual<const N: usize>` carries a value plus an N-wide sensitivity
//! vector. Every arithmetic op propagates derivatives using the elementary
//! chain rule. For a scalar function y = f(x_0, ..., x_{N-1}), evaluating f
//! on Duals where x_i has sensitivity e_i (the i-th unit vector) returns a
//! Dual whose value is y and whose sensitivities are (dy/dx_0, ..., dy/dx_{N-1}).
//!
//! For our problem (one scalar output per band, <= 14 inputs) forward-mode
//! is at parity with reverse-mode in performance and drastically simpler in
//! implementation — no tape, no heap, no lifetimes. The entire module
//! compiles to a ~50-line WASM binary contribution and auto-vectorises well.
//!
//! Usage:
//! ```ignore
//! use wall_tl_core::autodiff::Dual;
//! // Take gradient of y = x0^2 + 3*x1 at (x0, x1) = (4.0, 2.0).
//! let x0 = Dual::<2>::var(4.0, 0);
//! let x1 = Dual::<2>::var(2.0, 1);
//! let y = x0 * x0 + Dual::from(3.0) * x1;  // y.value = 22.0
//! assert_eq!(y.grad[0], 8.0);              // dy/dx0 = 2 x0 = 8
//! assert_eq!(y.grad[1], 3.0);              // dy/dx1 = 3
//! ```

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy)]
pub struct Dual<const N: usize> {
    pub value: f64,
    pub grad: [f64; N],
}

impl<const N: usize> Dual<N> {
    /// A constant with no sensitivity.
    pub const fn constant(value: f64) -> Self {
        Self {
            value,
            grad: [0.0; N],
        }
    }

    /// Declare a variable at slot `idx` — its own derivative is 1, others 0.
    pub fn var(value: f64, idx: usize) -> Self {
        let mut grad = [0.0; N];
        grad[idx] = 1.0;
        Self { value, grad }
    }

    pub fn powi(self, n: i32) -> Self {
        // d/dx x^n = n x^{n-1}
        let coeff = (n as f64) * self.value.powi(n - 1);
        Self {
            value: self.value.powi(n),
            grad: scale(&self.grad, coeff),
        }
    }

    pub fn sqrt(self) -> Self {
        let v = self.value.sqrt();
        // d sqrt / dx = 1 / (2 sqrt x)
        let coeff = 0.5 / v;
        Self {
            value: v,
            grad: scale(&self.grad, coeff),
        }
    }

    pub fn ln(self) -> Self {
        Self {
            value: self.value.ln(),
            grad: scale(&self.grad, 1.0 / self.value),
        }
    }

    pub fn log10(self) -> Self {
        // d log10(x) / dx = 1 / (x ln 10)
        Self {
            value: self.value.log10(),
            grad: scale(&self.grad, 1.0 / (self.value * std::f64::consts::LN_10)),
        }
    }

    pub fn log2(self) -> Self {
        Self {
            value: self.value.log2(),
            grad: scale(&self.grad, 1.0 / (self.value * std::f64::consts::LN_2)),
        }
    }

    pub fn exp(self) -> Self {
        let v = self.value.exp();
        Self {
            value: v,
            grad: scale(&self.grad, v),
        }
    }

    /// Sigmoid: 1 / (1 + e^{-x}). Derivative sigmoid * (1 - sigmoid).
    pub fn sigmoid(self) -> Self {
        let s = if self.value >= 0.0 {
            1.0 / (1.0 + (-self.value).exp())
        } else {
            let e = self.value.exp();
            e / (1.0 + e)
        };
        Self {
            value: s,
            grad: scale(&self.grad, s * (1.0 - s)),
        }
    }

    /// Smooth clamp? No — use `max` and `min` separately, which zero out
    /// gradients outside the active range (subgradient-at-0 convention).
    pub fn max_f(self, lo: f64) -> Self {
        if self.value >= lo {
            self
        } else {
            Self::constant(lo)
        }
    }

    pub fn min_f(self, hi: f64) -> Self {
        if self.value <= hi {
            self
        } else {
            Self::constant(hi)
        }
    }

    pub fn clamp_f(self, lo: f64, hi: f64) -> Self {
        self.max_f(lo).min_f(hi)
    }
}

fn scale<const N: usize>(g: &[f64; N], k: f64) -> [f64; N] {
    let mut out = [0.0; N];
    let mut i = 0;
    while i < N {
        out[i] = g[i] * k;
        i += 1;
    }
    out
}

fn combine<const N: usize>(a: &[f64; N], ka: f64, b: &[f64; N], kb: f64) -> [f64; N] {
    let mut out = [0.0; N];
    let mut i = 0;
    while i < N {
        out[i] = a[i] * ka + b[i] * kb;
        i += 1;
    }
    out
}

// ---- Arithmetic ----

impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            grad: combine(&self.grad, 1.0, &rhs.grad, 1.0),
        }
    }
}

impl<const N: usize> Sub for Dual<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            grad: combine(&self.grad, 1.0, &rhs.grad, -1.0),
        }
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        // d(uv)/dx = v du + u dv
        Self {
            value: self.value * rhs.value,
            grad: combine(&self.grad, rhs.value, &rhs.grad, self.value),
        }
    }
}

impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        // d(u/v)/dx = (v du - u dv) / v^2
        let inv_v = 1.0 / rhs.value;
        let value = self.value * inv_v;
        Self {
            value,
            grad: combine(&self.grad, inv_v, &rhs.grad, -value * inv_v),
        }
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            grad: scale(&self.grad, -1.0),
        }
    }
}

// --- scalar mixing ---

impl<const N: usize> Add<f64> for Dual<N> {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        Self {
            value: self.value + rhs,
            grad: self.grad,
        }
    }
}

impl<const N: usize> Sub<f64> for Dual<N> {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self {
        Self {
            value: self.value - rhs,
            grad: self.grad,
        }
    }
}

impl<const N: usize> Mul<f64> for Dual<N> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            value: self.value * rhs,
            grad: scale(&self.grad, rhs),
        }
    }
}

impl<const N: usize> Div<f64> for Dual<N> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        let inv = 1.0 / rhs;
        Self {
            value: self.value * inv,
            grad: scale(&self.grad, inv),
        }
    }
}

impl<const N: usize> From<f64> for Dual<N> {
    fn from(v: f64) -> Self {
        Self::constant(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Very simple: y = x0^2 + 3 x1, at (4, 2) -> y = 22, dy/dx0=8, dy/dx1=3.
    #[test]
    fn simple_polynomial() {
        let x0 = Dual::<2>::var(4.0, 0);
        let x1 = Dual::<2>::var(2.0, 1);
        let y = x0 * x0 + Dual::from(3.0) * x1;
        assert_relative_eq!(y.value, 22.0, max_relative = 1e-15);
        assert_relative_eq!(y.grad[0], 8.0, max_relative = 1e-15);
        assert_relative_eq!(y.grad[1], 3.0, max_relative = 1e-15);
    }

    /// Sanity: composite with sqrt, log, sigmoid. Finite-difference check.
    #[test]
    fn chain_rule_composite() {
        // y(a, b) = sigmoid(2 ln(a * b + 1) - 1) * sqrt(a^2 + b)
        let a = 1.5_f64;
        let b = 0.8_f64;
        let eval = |a: f64, b: f64| {
            let z = 2.0 * (a * b + 1.0).ln() - 1.0;
            let sig = if z >= 0.0 {
                1.0 / (1.0 + (-z).exp())
            } else {
                let e = z.exp();
                e / (1.0 + e)
            };
            sig * (a * a + b).sqrt()
        };
        let eps = 1e-7;
        let dfd_a = (eval(a + eps, b) - eval(a - eps, b)) / (2.0 * eps);
        let dfd_b = (eval(a, b + eps) - eval(a, b - eps)) / (2.0 * eps);

        let da = Dual::<2>::var(a, 0);
        let db = Dual::<2>::var(b, 1);
        let y = ((da * db + Dual::from(1.0)).ln() * Dual::from(2.0) - Dual::from(1.0)).sigmoid()
            * (da * da + db).sqrt();
        assert_relative_eq!(y.value, eval(a, b), max_relative = 1e-14);
        assert_relative_eq!(y.grad[0], dfd_a, max_relative = 1e-6);
        assert_relative_eq!(y.grad[1], dfd_b, max_relative = 1e-6);
    }

    /// Log-space gradient of the mass-law kernel matches the closed form.
    ///     TL(m) = 20 log10(pi f m / (rho0 c)) - 5
    ///     d TL / d ln m = 20 / ln 10 ≈ 8.686
    #[test]
    fn mass_law_via_duals_matches_closed_form() {
        let m = Dual::<1>::var(10.0, 0);
        let f = Dual::<1>::constant(500.0);
        let rho0 = Dual::<1>::constant(1.204);
        let c = Dual::<1>::constant(343.0);
        let pi = Dual::<1>::constant(std::f64::consts::PI);
        let ratio = pi * f * m / (rho0 * c);
        let tl = ratio.log10() * Dual::from(20.0) - Dual::from(5.0);
        // Linear-space gradient: m * d/dm TL should equal the log-space form.
        let ln_grad = tl.grad[0] * m.value;
        assert_relative_eq!(ln_grad, 20.0 / std::f64::consts::LN_10, max_relative = 1e-14);
    }
}
