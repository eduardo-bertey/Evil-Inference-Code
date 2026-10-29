/// MoSE — espejo de `moe-plus/moe.py::MoELayer` (solo inferencia).
/// Top-k sobre rutas (experto x ancho), expertos slimmable, shared experts.
use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::mlp::Mlp;

#[derive(Clone)]
pub struct ExpertSwiGLU {
    pub gate: candle_nn::Linear,
    pub up: candle_nn::Linear,
    pub down: candle_nn::Linear,
}

impl ExpertSwiGLU {
    pub fn new(vb: VarBuilder, d_model: usize, expert_dim: usize, bias: bool) -> Result<Self> {
        let (g, u, d) = if bias {
            (candle_nn::linear(d_model, expert_dim, vb.pp("w_gate"))?,
             candle_nn::linear(d_model, expert_dim, vb.pp("w_up"))?,
             candle_nn::linear(expert_dim, d_model, vb.pp("w_down"))?)
        } else {
            (candle_nn::linear_no_bias(d_model, expert_dim, vb.pp("w_gate"))?,
             candle_nn::linear_no_bias(d_model, expert_dim, vb.pp("w_up"))?,
             candle_nn::linear_no_bias(expert_dim, d_model, vb.pp("w_down"))?)
        };
        Ok(Self { gate: g, up: u, down: d })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(x)?;
        let up = self.up.forward(x)?;
        let h = (candle_nn::ops::silu(&gate)? * up)?;
        self.down.forward(&h)
    }
}

pub struct MoELayer {
    pub router: Linear,
    pub route_bias: Tensor,
    pub c_fc: Tensor,   // (E, D, 2*I)
    pub c_proj: Tensor, // (E, I, D)
    pub b_fc: Option<Tensor>,
    pub b_proj: Option<Tensor>,
    pub shared: Vec<ExpertSwiGLU>,
    pub n_experts: usize,
    pub top_k: usize,
    pub n_shared: usize,
    pub expert_dim: usize,
    pub widths: Vec<f32>,
    pub n_widths: usize,
    pub n_routes: usize,
    pub has_bias: bool,
}

impl MoELayer {
    pub fn new(
        vb: VarBuilder,
        d_model: usize,
        n_experts: usize,
        top_k: usize,
        n_shared: usize,
        expert_dim: usize,
        widths: Option<Vec<f32>>,
        bias: bool,
    ) -> Result<Self> {
        let widths = widths.unwrap_or_else(|| vec![0.25, 0.50, 0.75, 1.0]);
        let n_widths = widths.len();
        let n_routes = n_experts * n_widths;

        let router = candle_nn::linear_no_bias(d_model, n_routes, vb.pp("router"))?;
        let route_bias = vb.get(n_routes, "route_bias").unwrap_or_else(|_| {
            Tensor::zeros((n_routes,), DType::F32, vb.device()).unwrap()
        });

        // c_fc: (E, D, 2*I), c_proj: (E, I, D)
        let c_fc = vb.get((n_experts, d_model, 2 * expert_dim), "c_fc")?;
        let c_proj = vb.get((n_experts, expert_dim, d_model), "c_proj")?;
        let b_fc = if bias {
            Some(vb.get((n_experts, 2 * expert_dim), "b_fc")?)
        } else {
            None
        };
        let b_proj = if bias {
            Some(vb.get((n_experts, d_model), "b_proj")?)
        } else {
            None
        };

        let mut shared = Vec::with_capacity(n_shared);
        for i in 0..n_shared {
            shared.push(ExpertSwiGLU::new(vb.pp(format!("shared.{}", i)), d_model, expert_dim, bias)?);
        }

        Ok(Self {
            router, route_bias, c_fc, c_proj, b_fc, b_proj, shared,
            n_experts, top_k, n_shared, expert_dim, widths, n_widths, n_routes, has_bias: bias,
        })
    }

    /// forward -> (output, aux_loss). Inferencia: aux_loss = 0.
    pub fn forward(&self, x: &Tensor, width: Option<f32>) -> Result<(Tensor, f32)> {
        let (b, t, c) = x.dims3()?;
        let n = b * t;
        let flat = x.reshape((n, c))?;

        // router logits: (n, n_experts, n_widths)
        let logits = self.router.forward(&flat)?.reshape((n, self.n_experts, self.n_widths))?;
        let biased = logits.broadcast_add(&self.route_bias.reshape((1, self.n_experts, self.n_widths))?)?;

        // probs: (n, n_routes)
        let probs = if let Some(w) = width {
            // entrenamiento Eq.6: ancho global forzado
            let wi = self.widths.iter().enumerate()
                .min_by(|a, b| (a.1 - w).abs().partial_cmp(&(b.1 - w).abs()).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            let scores_e = biased.narrow(2, wi, 1)?.squeeze(2)?; // (n, E)
            candle_nn::ops::softmax(&scores_e, 1)?
        } else {
            candle_nn::ops::softmax(&biased.reshape((n, self.n_routes))?, 1)?
        };

        // top-k
        // manual: sort each row
        let probs_v = probs.to_vec2::<f32>()?;
        let mut topk_i = vec![0usize; n * self.top_k];
        let mut topk_w = vec![0.0f32; n * self.top_k];
        for i in 0..n {
            let mut idx: Vec<(usize, f32)> = probs_v[i].iter().copied().enumerate().collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut wsum = 0.0f32;
            for rank in 0..self.top_k {
                if rank < idx.len() {
                    topk_i[i * self.top_k + rank] = idx[rank].0;
                    topk_w[i * self.top_k + rank] = idx[rank].1;
                    wsum += idx[rank].1;
                }
            }
            // renormalize
            if wsum > 0.0 {
                for rank in 0..self.top_k {
                    topk_w[i * self.top_k + rank] /= wsum;
                }
            }
        }

        // capacity
        let cap = ((self.top_k * n) as f64 / self.n_routes as f64 * 1.35).ceil() as usize;
        let cap = cap.max(1);

        // execute routes
        let mut out = vec![0.0f32; n * c];
        // shared experts
        for sh in &self.shared {
            let y = sh.forward(&flat)?;
            let yv = y.to_vec2::<f32>()?;
            for i in 0..n {
                for j in 0..c {
                    out[i * c + j] += yv[i][j];
                }
            }
        }
        // routed
        for route in 0..self.n_routes {
            let e = route / self.n_widths;
            let wi = route % self.n_widths;
            // tokens where this route is in top-k
            let mut tok_list: Vec<usize> = Vec::new();
            for i in 0..n {
                for rank in 0..self.top_k {
                    if topk_i[i * self.top_k + rank] == route {
                        tok_list.push(i);
                    }
                }
            }
            if tok_list.is_empty() {
                continue;
            }
            // capacity trim by weight desc
            if tok_list.len() > cap {
                let mut pairs: Vec<(usize, f32)> = tok_list.iter().map(|&i| {
                    let mut w = 0.0f32;
                    for rank in 0..self.top_k {
                        if topk_i[i * self.top_k + rank] == route {
                            w = topk_w[i * self.top_k + rank];
                        }
                    }
                    (i, w)
                }).collect();
                pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                tok_list = pairs.iter().take(cap).map(|p| p.0).collect();
            }
            let run_w = if let Some(w) = width { w } else { self.widths[wi] };
            let d = ((self.expert_dim as f32 * run_w).round() as usize).max(8);
            // gather x for selected tokens
            let mut x_sel = vec![0.0f32; tok_list.len() * c];
            let mut w_sel = vec![0.0f32; tok_list.len()];
            for (k, &i) in tok_list.iter().enumerate() {
                for j in 0..c {
                    x_sel[k * c + j] = {
                        let v = flat.to_vec2::<f32>().unwrap();
                        v[i][j]
                    };
                }
                // weight
                let mut w = 0.0f32;
                for rank in 0..self.top_k {
                    if topk_i[i * self.top_k + rank] == route {
                        w = topk_w[i * self.top_k + rank];
                    }
                }
                w_sel[k] = w;
            }
            let x_sel_t = Tensor::from_vec(x_sel.clone(), (tok_list.len(), c), x.device())?;
            // slimmable forward
            // w_fc = c_fc[e][:, :2*d]  (D, 2d) ; x @ w_fc
            let c_fc_e = self.c_fc.get(e)?.contiguous()?; // (D, 2*I)
            let w_fc = c_fc_e.narrow(1, 0, 2 * d)?.contiguous()?; // (D, 2d)
            let c_proj_e = self.c_proj.get(e)?.contiguous()?; // (I, D)
            let w_proj = c_proj_e.narrow(0, 0, d)?.contiguous()?; // (d, D)

            // x_sel (Nsel, D) @ w_fc (D, 2d) -> (Nsel, 2d)
            let h = x_sel_t.matmul(&w_fc)?;
            // bias
            let h = if self.has_bias {
                let b_fc_e = self.b_fc.as_ref().unwrap().get(e)?;
                let b = Tensor::cat(&[b_fc_e.narrow(0, 0, d)?, b_fc_e.narrow(0, self.expert_dim, d)?], 0)?;
                h.broadcast_add(&b.reshape((1, 2 * d))?)?
            } else {
                h
            };
            // chunk gate/up
            let gate = h.narrow(1, 0, d)?;
            let up = h.narrow(1, d, d)?;
            let silu_gate = candle_nn::ops::silu(&gate)?;
            let h = (silu_gate * up)?; // (Nsel, d)
            // h @ w_proj (d, D) -> (Nsel, D)
            let expert_out = h.matmul(&w_proj)?;
            let expert_out = if self.has_bias {
                let b_proj_e = self.b_proj.as_ref().unwrap().get(e)?;
                expert_out.broadcast_add(&b_proj_e.reshape((1, c))?)?
            } else {
                expert_out
            };
            let eo = expert_out.to_vec2::<f32>()?;
            for (k, &i) in tok_list.iter().enumerate() {
                for j in 0..c {
                    out[i * c + j] += w_sel[k] * eo[k][j];
                }
            }
        }

        let out = Tensor::from_vec(out, (b, t, c), x.device())?;
        Ok((out, 0.0))
    }
}

pub struct DenseFFN {
    pub gate: candle_nn::Linear,
    pub up: candle_nn::Linear,
    pub down: candle_nn::Linear,
}

impl DenseFFN {
    pub fn new(vb: VarBuilder, dim: usize, hidden: usize, bias: bool) -> Result<Self> {
        let (g, u, d) = if bias {
            (candle_nn::linear(dim, hidden, vb.pp("gate_proj"))?,
             candle_nn::linear(dim, hidden, vb.pp("up_proj"))?,
             candle_nn::linear(hidden, dim, vb.pp("down_proj"))?)
        } else {
            (candle_nn::linear_no_bias(dim, hidden, vb.pp("gate_proj"))?,
             candle_nn::linear_no_bias(dim, hidden, vb.pp("up_proj"))?,
             candle_nn::linear_no_bias(hidden, dim, vb.pp("down_proj"))?)
        };
        Ok(Self { gate: g, up: u, down: d })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(x)?;
        let up = self.up.forward(x)?;
        let h = (candle_nn::ops::silu(&gate)? * up)?;
        self.down.forward(&h)
    }
}
