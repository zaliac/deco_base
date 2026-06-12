from yacs.config import CfgNode as CN

# Set default hparams to construct new default config
# Make sure the defaults are same as in parser
hparams = CN()

# General settings
hparams.EXP_NAME = 'default'
hparams.PROJECT_NAME = 'default'
hparams.OUTPUT_DIR = 'deco_results/'
hparams.CONDOR_DIR = '/is/cluster/work/achatterjee/condor/rich/'
hparams.LOGDIR = ''

# Dataset hparams
hparams.DATASET = CN()
hparams.DATASET.BATCH_SIZE = 64
hparams.DATASET.NUM_WORKERS = 4
hparams.DATASET.NORMALIZE_IMAGES = True

# Optimizer hparams
hparams.OPTIMIZER = CN()
hparams.OPTIMIZER.TYPE = 'adam'
hparams.OPTIMIZER.LR = 5e-5
hparams.OPTIMIZER.NUM_UPDATE_LR = 10

# Training hparams
hparams.TRAINING = CN()
hparams.TRAINING.ENCODER = 'hrnet'
hparams.TRAINING.CONTEXT = True
hparams.TRAINING.NUM_EPOCHS = 50
hparams.TRAINING.SUMMARY_STEPS = 100
hparams.TRAINING.CHECKPOINT_EPOCHS = 5
hparams.TRAINING.NUM_EARLY_STOP = 10
hparams.TRAINING.DATASETS = ['rich']
hparams.TRAINING.DATASET_MIX_PDF = ['1.']
hparams.TRAINING.DATASET_ROOT_PATH = '/is/cluster/work/achatterjee/rich/npzs'
hparams.TRAINING.BEST_MODEL_PATH = '/is/cluster/work/achatterjee/weights/rich/exp/rich_exp.pth'
hparams.TRAINING.LOSS_WEIGHTS = 1.
hparams.TRAINING.PAL_LOSS_WEIGHTS = 1.
hparams.TRAINING.DISTILL = False               # DINO teacher-student self-distillation (utils/distill.py)
hparams.TRAINING.DISTILL_DINO_WEIGHT = 1.0     # weight on the feature-DINO loss
hparams.TRAINING.DISTILL_OUT_WEIGHT = 1.0      # weight on the output-consistency loss
hparams.TRAINING.DISTILL_ATTN_WEIGHT = 0.0     # weight on the fusion attention-consistency loss (0 = off)
hparams.TRAINING.DISTILL_OUT_DIM = 4096        # DINO prototype count
hparams.TRAINING.EMA_MOMENTUM = 0.996          # teacher EMA momentum
hparams.TRAINING.DINO_TEACHER_TEMP = 0.04
hparams.TRAINING.DINO_STUDENT_TEMP = 0.1
hparams.TRAINING.DINO_CENTER_MOMENTUM = 0.9
# Self-supervised task on the DINOv3 ViT backbone's self-attention (uses Q,K,V). Requires
# BACKBONE_UNFREEZE_N>0 (gradients must reach the qkv proj) and DISTILL=True (reuses the
# two-view EMA-teacher pipeline). See utils/distill.py:tap_dinov3_attention.
hparams.TRAINING.BACKBONE_UNFREEZE_N = 0             # fine-tune the top-N backbone blocks (0 = fully frozen)
hparams.TRAINING.DISTILL_BACKBONE_ATTN_WEIGHT = 0.0  # cross-view consistency on the attention OUTPUT softmax(QKᵀ)V (uses Q,K,V); 0 = off
hparams.TRAINING.DISTILL_BACKBONE_MAP_WEIGHT = 0.0   # cross-view KL on the explicit QKᵀ attention MAP (uses Q,K); 0 = off (higher memory)

# DINO teacher-student TEST-TIME ADAPTATION (utils/dino_tta.py), used in tester.py only.
# Frozen-anchor, episodic: per test batch adapt a small downstream adapt-set to match a fixed
# teacher across two photometric views, then predict + reset. IN-DOMAIN this is ~a no-op (the
# model is already augmentation-consistent); the step-0 consistency it prints is the no-op tell.
hparams.TRAINING.DINO_TTA = False              # turn on DINO test-time adaptation in tester.py
hparams.TRAINING.DINO_TTA_STEPS = 2            # inner SGD steps per test batch (0 = plain predict)
hparams.TRAINING.DINO_TTA_LR = 1e-3            # inner-loop SGD lr (small -> episodic stability)
hparams.TRAINING.DINO_TTA_OUT_WEIGHT = 1.0     # weight on output (contact-prob) consistency MSE
hparams.TRAINING.DINO_TTA_FEAT_WEIGHT = 1.0    # weight on pooled-feature cosine consistency
hparams.TRAINING.DINO_TTA_ONLINE = False       # True = keep adapted weights across batches (EMA teacher, CoTTA-style)

# Training hparams
hparams.VALIDATION = CN()
hparams.VALIDATION.SUMMARY_STEPS = 100
hparams.VALIDATION.DATASETS = ['rich']
hparams.VALIDATION.MAIN_DATASET = 'rich'
