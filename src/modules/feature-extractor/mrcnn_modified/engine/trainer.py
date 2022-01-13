# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from mrcnn_modified.data import make_data_loader
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from mrcnn_modified.engine.inference_full_mask import inference

from apex import amp

from maskrcnn_benchmark.utils.model_serialization import load_state_dict


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    is_target_task=False,
    icwt_21_objs=False,
    training_seconds=None,
    model_val = None,
    checkpointer_val=None
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        
        if any(len(target) < 1 for target in targets):
            logger.error("Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if training_seconds:
            if end - start_training_time > training_seconds:
                if data_loader_val is not None:
                    meters_val = MetricLogger(delimiter="  ")
                    synchronize()
                    if model_val:
                        load_state_dict(checkpointer_val.model, checkpointer.model.state_dict())
                        model = checkpointer_val.model
                    _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                        model,
                        # The method changes the segmentation mask format in a data loader,
                        # so every time a new data loader is created:
                        make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_target_task=is_target_task, icwt_21_objs=icwt_21_objs),  # is_for_period=True),
                        dataset_name="[Test]",
                        iou_types=iou_types,
                        box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                        device=cfg.MODEL.DEVICE,
                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        output_folder=None,
                        is_target_task=is_target_task,
                        icwt_21_objs=icwt_21_objs
                    )
                    synchronize()
                    model.train()
                    with torch.no_grad():
                        # Should be one image for each GPU:
                        for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                            images_val = images_val.to(device)
                            targets_val = [target.to(device) for target in targets_val]
                            loss_dict = model(images_val, targets_val)
                            losses = sum(loss for loss in loss_dict.values())
                            loss_dict_reduced = reduce_loss_dict(loss_dict)
                            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                            meters_val.update(loss=losses_reduced, **loss_dict_reduced)
                    synchronize()
                    logger.info(
                        meters_val.delimiter.join(
                            [
                                "[Test]: ",
                                "eta: {eta}",
                                "iter: {iter}",
                                "{meters}",
                                "lr: {lr:.6f}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            eta=eta_string,
                            iter=iteration,
                            meters=str(meters_val),
                            lr=optimizer.param_groups[0]["lr"],
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )
                checkpointer.save("model_final", **arguments)
                break


        else:
            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
                meters_val = MetricLogger(delimiter="  ")
                synchronize()
                if model_val:
                    load_state_dict(checkpointer_val.model, checkpointer.model.state_dict())
                    model = checkpointer_val.model
                _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                    model,
                    # The method changes the segmentation mask format in a data loader,
                    # so every time a new data loader is created:
                    make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_target_task=is_target_task, icwt_21_objs=icwt_21_objs),# is_for_period=True),
                    dataset_name="[Test]",
                    iou_types=iou_types,
                    box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                    device=cfg.MODEL.DEVICE,
                    expected_results=cfg.TEST.EXPECTED_RESULTS,
                    expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                    output_folder=None,
                    is_target_task=is_target_task,
                    icwt_21_objs=icwt_21_objs
                )
                synchronize()
                model.train()
                with torch.no_grad():
                    # Should be one image for each GPU:
                    for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                        images_val = images_val.to(device)
                        targets_val = [target.to(device) for target in targets_val]
                        loss_dict = model(images_val, targets_val)
                        losses = sum(loss for loss in loss_dict.values())
                        loss_dict_reduced = reduce_loss_dict(loss_dict)
                        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                        meters_val.update(loss=losses_reduced, **loss_dict_reduced)
                synchronize()
                logger.info(
                    meters_val.delimiter.join(
                        [
                            "[Test]: ",
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters_val),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
                if model_val:
                    #Restore the old model to restart the training
                    load_state_dict(checkpointer.model, checkpointer_val.model.state_dict())
                    model = checkpointer.model
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
