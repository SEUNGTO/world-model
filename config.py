# Hyperparameters
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2  # Accumulate 2 steps to simulate batch_size=32
EPOCHS = 20
LR = 1e-4

# Feature Size
LATENT_DIM       = 2**8    # Back to 256 - 40GB can handle full model
FUSED_DIM        = 2**8    # Back to 256 - 40GB can handle full model
TICK_FEAT_DIM    = 11     # tick feature dimension (from build_dataset)
MAX_OBS_TICKS    = 2**10  # Keep at 4096 to match preprocessed data
MAX_TARGET_TICKS = 2**10  # Keep at 4096 to match preprocessed data

DIFFUSION_STEPS = 500
SAMPLE_STEPS = 100

# Data building
minutes = 10
chunk_size = 1e6
