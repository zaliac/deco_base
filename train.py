import torch
from torch.utils.data import DataLoader
import os

from train.trainer_step import TrainStepper
from train.base_trainer import trainer, evaluator
from data.base_dataset import BaseDataset
from data.mixed_dataset import MixedDataset
from models.deco import DECO
from utils.config import parse_args, run_grid_search_experiments

# add imports
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def train(hparams):
    deco_model = DECO(hparams.TRAINING.ENCODER, hparams.TRAINING.CONTEXT, device)

    solver = TrainStepper(deco_model, hparams.TRAINING.CONTEXT, hparams.OPTIMIZER.LR, hparams.TRAINING.LOSS_WEIGHTS, hparams.TRAINING.PAL_LOSS_WEIGHTS, device)

    # create tensorboard writer
    tb_log_dir = os.path.join(hparams.OUTPUT_DIR, hparams.EXP_NAME, 'tb_logs',
                              datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # pass writer into TrainStepper (if you kept TrainStepper unchanged, pass writer to trainer/evaluator calls later)
    # if you choose to attach writer to solver, modify TrainStepper __init__ signature to accept writer and set self.writer = writer

    vb_f1 = 0
    start_ep = 0
    num = 0
    k = True
    latest_model_path = hparams.TRAINING.BEST_MODEL_PATH.replace('best', 'latest')
    if os.path.exists(latest_model_path):
      _, vb_f1 = solver.load(hparams.TRAINING.BEST_MODEL_PATH)
      start_ep, _ = solver.load(latest_model_path)
    
    if getattr(hparams.TRAINING, 'DISTILL', False):
        solver.enable_distill(out_dim=hparams.TRAINING.DISTILL_OUT_DIM,
                              dino_weight=hparams.TRAINING.DISTILL_DINO_WEIGHT,
                              out_weight=hparams.TRAINING.DISTILL_OUT_WEIGHT,
                              ema_momentum=hparams.TRAINING.EMA_MOMENTUM,
                              teacher_temp=hparams.TRAINING.DINO_TEACHER_TEMP,
                              student_temp=hparams.TRAINING.DINO_STUDENT_TEMP,
                              center_momentum=hparams.TRAINING.DINO_CENTER_MOMENTUM,
                              ramp_steps=hparams.TRAINING.DISTILL_RAMP_STEPS)

    for epoch in range(start_ep+1, hparams.TRAINING.NUM_EPOCHS + 1):
        # Train one epoch
        # trainer(epoch, train_loader, solver, hparams)
        trainer(epoch, train_loader, solver, hparams, writer=writer)
        # Run evaluation
        vc_f1 = None
        for val_loader in val_loaders:
            dataset_name = val_loader.dataset.dataset
            # vc_f1_ds = evaluator(val_loader, solver, hparams, epoch, dataset_name, normalize=hparams.DATASET.NORMALIZE_IMAGES)
            vc_f1_ds = evaluator(val_loader, solver, hparams, epoch, dataset_name,
                                 normalize=hparams.DATASET.NORMALIZE_IMAGES,
                                 writer=writer)
            if dataset_name == hparams.VALIDATION.MAIN_DATASET:
                vc_f1 = vc_f1_ds
        if vc_f1 is None:
            raise ValueError('Main dataset not found in validation datasets')

        print('Learning rate: ', solver.lr)

        print('---------------------------------------------')
        print('---------------------------------------------')

        solver.save(epoch, vc_f1, latest_model_path)

        if epoch % hparams.TRAINING.CHECKPOINT_EPOCHS == 0:
          inter_model_path = latest_model_path.replace('latest', 'epoch_'+str(epoch).zfill(3))
          solver.save(epoch, vc_f1, inter_model_path)

        if vc_f1 < vb_f1:
          num += 1
          print('Not Saving model: Best Val F1 = ', vb_f1, ' Current Val F1 = ', vc_f1)
        else:
          num = 0
          vb_f1 = vc_f1
          print('Saving model...')
          solver.save(epoch, vb_f1, hparams.TRAINING.BEST_MODEL_PATH)

        if num >= hparams.OPTIMIZER.NUM_UPDATE_LR: solver.update_lr()
        if num >= hparams.TRAINING.NUM_EARLY_STOP:
          print('Early Stop')
          k = False

        if k: continue
        else: break
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    hparams = run_grid_search_experiments(
        args,
        script='train.py',
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset_root_path = hparams.TRAINING.DATASET_ROOT_PATH

    train_dataset = MixedDataset(hparams.TRAINING.DATASETS, 'train', dataset_mix_pdf=hparams.TRAINING.DATASET_MIX_PDF, dataset_root_path=dataset_root_path, normalize=hparams.DATASET.NORMALIZE_IMAGES)

    val_datasets = []
    for ds in hparams.VALIDATION.DATASETS:
        if ds in ['rich', 'prox']:
            val_datasets.append(BaseDataset(ds, 'val', model_type='smplx', dataset_root_path=dataset_root_path, normalize=hparams.DATASET.NORMALIZE_IMAGES))
        elif ds in ['damon']:
            val_datasets.append(BaseDataset(ds, 'val', model_type='smpl', dataset_root_path=dataset_root_path, normalize=hparams.DATASET.NORMALIZE_IMAGES))
        else:
            raise ValueError('Dataset not supported')

    train_loader = DataLoader(train_dataset, hparams.DATASET.BATCH_SIZE, shuffle=True, num_workers=hparams.DATASET.NUM_WORKERS)
    val_loaders = [DataLoader(val_dataset, batch_size=hparams.DATASET.BATCH_SIZE, shuffle=False, num_workers=hparams.DATASET.NUM_WORKERS) for val_dataset in val_datasets]

    train(hparams)


# tensorboard --logdir deco_results/demo_train/tb_logs
# # If you used a timestamped folder, point to deco_results/demo_train/tb_logs