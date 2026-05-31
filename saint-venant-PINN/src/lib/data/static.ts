export const TRUE_PARAMS = {
  n: 0.035,
  S0: 0.001,
  Q0: 50,
  B_w: 50,
  L: 5000,
} as const

export const PINN_CONFIG = {
  activation: "tanh",
  normalization: "Q / Q_max",
  hidden_size: 64,
  n_layers: 4,
  N_EPOCHS_ADAM: 6000,
  N_EPOCHS_WARMUP: 2000,
  N_EPOCHS_RAMP: 1000,
  N_ITER_LBFGS: 500,
  LAMBDA_DATA: 1.0,
  LAMBDA_PDE: 0.05,
  gradient_clip_max_norm: 1.0,
  N_COLLOC: 2000,
  WARMUP_H: 1.0,
} as const
