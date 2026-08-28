import torch
from torch.utils.data import DataLoader
from loguru import logger

from train.trainer_step import TrainStepper
from train.base_trainer import evaluator
from data.base_dataset import BaseDataset
from models.deco import DECO
from utils.config import parse_args, run_grid_search_experiments


def assert_checkpoint_encoder_compatible(model_path, encoder_type):
    """Reject the semantic-encoder mismatch that silently ruins evaluation."""
    expected_prefix = {
        'sam_hrnet': 'encoder_sem.encoder.',
        'sam_sam': 'encoder_sem.backbone.',
    }.get(encoder_type)
    if expected_prefix is None:
        return

    try:
        checkpoint = torch.load(
            model_path, map_location='meta', mmap=True, weights_only=False,
        )
    except (TypeError, ValueError):
        checkpoint = torch.load(model_path, map_location='meta', weights_only=False)
    state_dict = checkpoint.get('deco', checkpoint)
    if not any(name.startswith(expected_prefix) for name in state_dict):
        found = 'SAM-3D-Objects' if any(
            name.startswith('encoder_sem.backbone.') for name in state_dict
        ) else 'HRNet/other'
        expected = 'SAM-3D-Objects' if encoder_type == 'sam_sam' else 'HRNet'
        raise RuntimeError(
            f'Checkpoint {model_path} has a {found} semantic encoder, but '
            f'TRAINING.ENCODER={encoder_type!r} requires {expected}. '
            'Use a checkpoint trained with the selected encoder.'
        )


def test(hparams):
    deco_model = DECO(hparams.TRAINING.ENCODER, hparams.TRAINING.CONTEXT, device)
    pytorch_total_params = sum(p.numel() for p in deco_model.parameters() if p.requires_grad)
    print('Total number of trainable parameters: ', pytorch_total_params)

    solver = TrainStepper(deco_model, hparams.TRAINING.CONTEXT, hparams.OPTIMIZER.LR, hparams.TRAINING.LOSS_WEIGHTS, hparams.TRAINING.PAL_LOSS_WEIGHTS, device)

    assert_checkpoint_encoder_compatible(
        hparams.TRAINING.BEST_MODEL_PATH, hparams.TRAINING.ENCODER,
    )
    logger.info(f'Loading weights from {hparams.TRAINING.BEST_MODEL_PATH}')
    _, _ = solver.load(hparams.TRAINING.BEST_MODEL_PATH)
    if hparams.TEST_TIME.MULTIVIEW_ENABLED:
        solver.enable_multiview_inference(
            original_weight=hparams.TEST_TIME.ENSEMBLE_ORIGINAL_WEIGHT,
            trim_fraction=hparams.TEST_TIME.ENSEMBLE_TRIM_FRACTION,
            min_positive_iou=hparams.TEST_TIME.MULTIVIEW_MIN_POSITIVE_IOU,
            min_agreeing_frames=hparams.TEST_TIME.MULTIVIEW_MIN_AGREEING_FRAMES,
        )
        logger.info(
            'Task-7 multi-view inference enabled: original + '
            f'{hparams.TEST_TIME.NUM_FRAMES} rendered views; '
            f'original logit weight={hparams.TEST_TIME.ENSEMBLE_ORIGINAL_WEIGHT:g}, '
            f'trim={hparams.TEST_TIME.ENSEMBLE_TRIM_FRACTION:g}, '
            f'min positive IoU={hparams.TEST_TIME.MULTIVIEW_MIN_POSITIVE_IOU:g}, '
            f'min agreeing views={hparams.TEST_TIME.MULTIVIEW_MIN_AGREEING_FRAMES}'
        )
    
    # Run testing
    for test_loader in val_loaders:
        dataset_name = test_loader.dataset.dataset
        test_dict, total_time = evaluator(test_loader, solver, hparams, 0, dataset_name, return_dict=True)

        print('Test Contact Precision: ', test_dict['cont_precision'])
        print('Test Contact Recall: ', test_dict['cont_recall'])
        print('Test Contact F1 Score: ', test_dict['cont_f1'])
        print('Test Contact F1 Score (paper: harmonic of mean P/R): ', test_dict['cont_f1_paper'])
        print('Test Contact FP Geo. Error: ', test_dict['fp_geo_err'])
        print('Test Contact FN Geo. Error: ', test_dict['fn_geo_err'])
        if hparams.TRAINING.CONTEXT:
            print('Test Contact Semantic Segmentation IoU: ', test_dict['sem_iou'])
            print('Test Contact Part Segmentation IoU: ', test_dict['part_iou'])
        print('\nTime taken per image for evaluation: ', total_time)
        print('-'*50)

if __name__ == '__main__':
    args = parse_args()
    hparams = run_grid_search_experiments(
        args,
        script='tester.py',
        change_wt_name=False
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    val_datasets = []
    use_sam_object_masks = hparams.TRAINING.ENCODER == 'sam_sam'
    dataset_root_path = hparams.TRAINING.DATASET_ROOT_PATH
    sam_object_mask_kwargs = {
        'generate_object_masks': use_sam_object_masks,
        'sam_keypoint_circle_radius': hparams.TRAINING.SAM_KEYPOINT_CIRCLE_RADIUS,
        'sam_keypoint_circle_points': hparams.TRAINING.SAM_KEYPOINT_CIRCLE_POINTS,
    }
    for ds in hparams.VALIDATION.DATASETS:
        multiview_kwargs = {
            'multiview_frames_root': (
                hparams.TEST_TIME.FRAMES_ROOT
                if hparams.TEST_TIME.MULTIVIEW_ENABLED and ds == 'damon' else None
            ),
            'multiview_num_frames': hparams.TEST_TIME.NUM_FRAMES,
        }
        if ds in ['rich', 'prox']:
            val_datasets.append(BaseDataset(
                ds, 'val', model_type='smplx', dataset_root_path=dataset_root_path,
                normalize=hparams.DATASET.NORMALIZE_IMAGES,
                **multiview_kwargs,
                **sam_object_mask_kwargs,
            ))
        elif ds in ['damon', 'behave']:
            val_datasets.append(BaseDataset(
                ds, 'val', model_type='smpl', dataset_root_path=dataset_root_path,
                normalize=hparams.DATASET.NORMALIZE_IMAGES,
                **multiview_kwargs,
                **sam_object_mask_kwargs,
            ))
        else:
            raise ValueError('Dataset not supported')

    # Dynamic SAM inference uses CUDA and therefore cannot safely occur in the
    # default forked DataLoader workers after the model has been initialized.
    # Twenty-one 256px float views per sample exceed the small shared-memory
    # budget commonly available to worker processes.  Load them in the main
    # process when Task-7 inference is on; each view is then transferred to GPU
    # one at a time by TrainStepper.evaluate.
    data_workers = 0 if (use_sam_object_masks or hparams.TEST_TIME.MULTIVIEW_ENABLED) else hparams.DATASET.NUM_WORKERS
    val_loaders = [
        DataLoader(
            val_dataset, batch_size=hparams.DATASET.BATCH_SIZE, shuffle=False,
            num_workers=data_workers,
        )
        for val_dataset in val_datasets
    ]

    test(hparams)
