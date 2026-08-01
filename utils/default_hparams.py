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
hparams.TRAINING.DATASET_ROOT_PATH = '/home/l_z80934/scratch/RICH'
hparams.TRAINING.BEST_MODEL_PATH = '/is/cluster/work/achatterjee/weights/rich/exp/rich_exp.pth'
hparams.TRAINING.LOSS_WEIGHTS = 1.
hparams.TRAINING.PAL_LOSS_WEIGHTS = 1.
hparams.TRAINING.DISTILL = False               # DINO teacher-student self-distillation (utils/distill.py)
hparams.TRAINING.DISTILL_DINO_WEIGHT = 0.1     # feature-DINO prototype CE (the named DINO method)
hparams.TRAINING.DISTILL_OUT_WEIGHT = 1.0      # output-consistency MSE on contact probs (on-task, primary)
hparams.TRAINING.DISTILL_OUT_DIM = 4096        # DINO prototype count
hparams.TRAINING.EMA_MOMENTUM = 0.996          # teacher EMA momentum
hparams.TRAINING.DINO_TEACHER_TEMP = 0.04
hparams.TRAINING.DINO_STUDENT_TEMP = 0.1
hparams.TRAINING.DINO_CENTER_MOMENTUM = 0.9
hparams.TRAINING.DISTILL_RAMP_STEPS = 2000     # warm distill terms 0->1 over this many batches
# Task 8: train the contact path on conservative centred zooms before using
# scale TTA.  Disabled globally to preserve existing training recipes.
hparams.TRAINING.DISTILL_SCALE_AUG_ENABLED = False
hparams.TRAINING.DISTILL_SCALE_AUG_PROBABILITY = 0.5
hparams.TRAINING.DISTILL_SCALE_AUG_MIN = 1.02
hparams.TRAINING.DISTILL_SCALE_AUG_MAX = 1.15
hparams.TRAINING.DISTILL_SCALE_CONSISTENCY_WEIGHT = 0.25
hparams.TRAINING.DISTILL_SCALE_AUG_MIN_KEYPOINT_RETENTION = 0.8
hparams.TRAINING.DISTILL_SCALE_AUG_MIN_OBJECT_RETENTION = 0.8
hparams.TRAINING.DISTILL_SCALE_AUG_FOCUS_PROMPTS = True
hparams.TRAINING.SAM_KEYPOINT_CIRCLE_RADIUS = 12.0
hparams.TRAINING.SAM_KEYPOINT_CIRCLE_POINTS = 8

# Per-image, label-free test-time adaptation (Task 7).  The original crop is a
# teacher target for centred zoom views; final prediction averages all scales.
# Disabled by default so validation remains checkpoint-only unless requested.
hparams.TEST_TIME = CN()
hparams.TEST_TIME.ENABLED = False
hparams.TEST_TIME.STEPS = 2
hparams.TEST_TIME.LR = 1e-5
hparams.TEST_TIME.ZOOM_SCALES = [1.25, 1.5]
hparams.TEST_TIME.CONSISTENCY_WEIGHT = 1.0
hparams.TEST_TIME.ENSEMBLE_ORIGINAL_WEIGHT = 1.0
# Keep the adapted original crop close to its checkpoint prediction.  This is
# particularly important for per-image adaptation, where one update can
# otherwise degrade the most reliable (unzoomed) view.
hparams.TEST_TIME.ORIGINAL_ANCHOR_WEIGHT = 1.0
hparams.TEST_TIME.HIGHRES_ZOOM_ENABLED = True
hparams.TEST_TIME.HIGHRES_SIZE = 512
hparams.TEST_TIME.MIN_KEYPOINT_RETENTION = 0.8
hparams.TEST_TIME.MIN_OBJECT_RETENTION = 0.8
# Select this only on a held-out validation split, never on the reported test set.
hparams.TEST_TIME.CONTACT_THRESHOLD = 0.5
hparams.TEST_TIME.EMA_MOMENTUM = 0.996
hparams.TEST_TIME.FOCUS_PROMPTS = True

# Training hparams
hparams.VALIDATION = CN()
hparams.VALIDATION.SUMMARY_STEPS = 100
hparams.VALIDATION.DATASETS = ['rich']
hparams.VALIDATION.MAIN_DATASET = 'rich'
