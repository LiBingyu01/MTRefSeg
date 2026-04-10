# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

import torch
import time
import tqdm
from utils import AverageMeter, ProgressMeter, Summary
import numpy as np

def train_one_epoch(train_loader, model_engine, epoch, train_iter, args, logger):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")


    progress = ProgressMeter(
        len(train_loader) if args.no_sampling else args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs),
        logger=logger
    )

    # switch to train mode
    model_engine.train()
    end = time.time()
    
    
    # [修改: 提取统一的精度 dtype]
    if args.precision == "fp16":
        dtype = torch.half
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float
        
    if args.no_sampling:
        for global_step, input_dict in enumerate(train_loader):
            
            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            # [修改: 兼容双时相图像的精度转换]
            if "images_t1" in input_dict:
                input_dict["images_t1"] = input_dict["images_t1"].to(dtype)
                input_dict["images_t2"] = input_dict["images_t2"].to(dtype)
                input_dict["images_clip_t1"] = input_dict["images_clip_t1"].to(dtype)
                input_dict["images_clip_t2"] = input_dict["images_clip_t2"].to(dtype)
                batch_size = input_dict["images_t1"].size(0)
            else:
                input_dict["images"] = input_dict["images"].to(dtype)
                input_dict["images_clip"] = input_dict["images_clip"].to(dtype)
                batch_size = input_dict["images"].size(0)
                
            output_dict = model_engine(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            mask_obj_loss = output_dict.get("mask_obj_loss", torch.zeros_like(mask_loss))
            
            
            # [修改: 使用动态获取的 batch_size]
            losses.update(loss.item(), batch_size)
            ce_losses.update(ce_loss.item(), batch_size)
            mask_bce_losses.update(mask_bce_loss.item(), batch_size)
            mask_dice_losses.update(mask_dice_loss.item(), batch_size)
            mask_losses.update(mask_loss.item(), batch_size)
            model_engine.backward(loss)
            model_engine.step()
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % args.print_freq == 0:
                if args.distributed:
                    batch_time.all_reduce()
                    data_time.all_reduce()
                    losses.all_reduce()
                    ce_losses.all_reduce()
                    mask_bce_losses.all_reduce()
                    mask_dice_losses.all_reduce()
                    mask_losses.all_reduce()

                if args.rank == 0:
                    progress.display(1 + global_step)
                    
                batch_time.reset()
                data_time.reset()
                losses.reset()
                ce_losses.reset()
                mask_bce_losses.reset()
                mask_dice_losses.reset()
                mask_losses.reset()

        return train_iter
    else:
        for global_step in range(args.steps_per_epoch):
            for i in range(args.grad_accumulation_steps):
                try:
                    input_dict = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    input_dict = next(train_iter)

                data_time.update(time.time() - end)
                input_dict = dict_to_cuda(input_dict)

                # [修改: 兼容双时相图像的精度转换]
                if "images_t1" in input_dict:
                    input_dict["images_t1"] = input_dict["images_t1"].to(dtype)
                    input_dict["images_t2"] = input_dict["images_t2"].to(dtype)
                    input_dict["images_clip_t1"] = input_dict["images_clip_t1"].to(dtype)
                    input_dict["images_clip_t2"] = input_dict["images_clip_t2"].to(dtype)
                    batch_size = input_dict["images_t1"].size(0)
                else:
                    input_dict["images"] = input_dict["images"].to(dtype)
                    input_dict["images_clip"] = input_dict["images_clip"].to(dtype)
                    batch_size = input_dict["images"].size(0)

                output_dict = model_engine(**input_dict)

                loss = output_dict["loss"]
                ce_loss = output_dict["ce_loss"]
                mask_bce_loss = output_dict["mask_bce_loss"]
                mask_dice_loss = output_dict["mask_dice_loss"]
                mask_loss = output_dict["mask_loss"]

                # [修改: 使用动态获取的 batch_size]
                losses.update(loss.item(), batch_size)
                ce_losses.update(ce_loss.item(), batch_size)
                mask_bce_losses.update(mask_bce_loss.item(), batch_size)
                mask_dice_losses.update(mask_dice_loss.item(), batch_size)
                mask_losses.update(mask_loss.item(), batch_size)

                model_engine.backward(loss)
            
                model_engine.step()
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % args.print_freq == 0:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

                if args.rank == 0:
                    progress.display(1 + global_step)
                    
                batch_time.reset()
                data_time.reset()
                losses.reset()
                ce_losses.reset()
                mask_bce_losses.reset()
                mask_dice_losses.reset()
                mask_losses.reset()

        return train_iter

@torch.no_grad()
def validate(val_loader, model_engine, epoch, logger, args): # 修改: 将 writer 改为 logger
    model_engine.eval()
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    # 记录所有图片的 IoU，用于最后求 Mean IoU
    all_ious = []
    cum_I, cum_U = 0.0, 0.0
    # --- 2. 遍历验证集 ---
    for input_dict in tqdm.tqdm(val_loader, desc=f"Validating Epoch {epoch}"):
        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            dtype = torch.half
        elif args.precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float
        if "images_t1" in input_dict:
            input_dict["images_t1"] = input_dict["images_t1"].to(dtype)
            input_dict["images_t2"] = input_dict["images_t2"].to(dtype)
            input_dict["images_clip_t1"] = input_dict["images_clip_t1"].to(dtype)
            input_dict["images_clip_t2"] = input_dict["images_clip_t2"].to(dtype)
        else:
            input_dict["images"] = input_dict["images"].to(dtype)
            input_dict["images_clip"] = input_dict["images_clip"].to(dtype)
        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        # --- 3. 获取预测和真值 ---
        pred_masks = output_dict["pred_masks"][0]
        gt_masks = output_dict["gt_masks"][0].int()

        # 统一维度为 [B, H, W]
        if pred_masks.shape[1] == 1:
            pred_masks = pred_masks.squeeze(1)
        
        # 二值化处理
        pred_prob = torch.sigmoid(pred_masks)
        pred_bin = (pred_prob > 0.5).float()
        gt_bin = gt_masks.float()

        # 将 [B, H, W] 展平为 [B, H*W]，方便在维度 1 上求和
        batch_size = pred_bin.shape[0]
        pred_flat = pred_bin.view(batch_size, -1)
        gt_flat = gt_bin.view(batch_size, -1)

        # 计算每张图的 Intersection (I) 和 Union (U)
        # sum(dim=1) 会返回一个 shape 为 [B] 的 tensor，包含这个 batch 中每张图的数值
        intersection_tensor = (pred_flat * gt_flat).sum(dim=1) # 交集 ONLY foreground 
        union_tensor = (pred_flat + gt_flat).sum(dim=1) - intersection_tensor # 并集

        # 防止除零，计算每张图的 IoU
        # iou_tensor shape: [B]
        iou_tensor = intersection_tensor / (union_tensor + 1e-6)

        # --- 5. 更新统计指标 ---

        # A. 更新 Overall IoU 的累积值 (累计整个数据集的 I 和 U)
        cum_I += intersection_tensor.sum().item()
        cum_U += union_tensor.sum().item()

        # B. 更新 Mean IoU 列表 (转为 numpy 存入 list)
        current_ious = iou_tensor.cpu().numpy()
        all_ious.extend(current_ious)

        # C. 更新 Precision@k (统计有多少张图的 IoU 超过了阈值)
        # 利用广播机制比较: [B, 1] >= [1, 5] -> [B, 5]
        # 然后在 batch 维度求和 -> [5]
        matches = (current_ious[:, None] >= np.array(eval_seg_iou_list)[None, :])
        seg_correct += matches.sum(axis=0)
        
        seg_total += batch_size

    # --- 6. 最终结果汇总与打印 ---
    if args.local_rank == 0:
        if len(all_ious) > 0:
            mIoU = np.mean(all_ious)
        else:
            mIoU = 0.0

        if cum_U > 0:
            overall_IoU = cum_I / cum_U
        else:
            overall_IoU = 0.0

        # 修改: 使用 logger.info 输出验证结果，并移除 writer.add_scalar
        logger.info('\n' + '=' * 40)
        logger.info(f'Validation Results (Epoch {epoch}):')
        logger.info('=' * 40)
        logger.info('Mean IoU (mIoU):       %.2f' % (mIoU * 100.))
        logger.info('Overall IoU (oIoU):    %.2f' % (overall_IoU * 100.))
        logger.info('-' * 40)
        
        results_str = ''
        for n_eval_iou in range(len(eval_seg_iou_list)):
            if seg_total > 0:
                res = seg_correct[n_eval_iou] * 100. / seg_total
            else:
                res = 0
            results_str += 'Precision @ %.1f:       %.2f%%\n' % \
                           (eval_seg_iou_list[n_eval_iou], res)
        
        # 防止 logger 处理多行字符串时格式错乱，可以按行输出，或者保持 \n
        logger.info(results_str)
        logger.info('=' * 40 + '\n')

        return mIoU * 100., overall_IoU * 100.
    
    else:
        # 非主进程返回 0
        return 0.0, 0.0

@torch.no_grad()
def eval_gres(val_loader, model_engine, epoch, args, logger):
    model_engine.eval()
    inter_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    g_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    nt_tp_meter = AverageMeter("NT_TP", ":6.3f", Summary.SUM)
    nt_tn_meter = AverageMeter("NT_TN", ":6.3f", Summary.SUM)
    nt_fp_meter = AverageMeter("NT_FP", ":6.3f", Summary.SUM)
    nt_fn_meter = AverageMeter("NT_FN", ":6.3f", Summary.SUM)
    is_grefcoco = val_loader.dataset.ds == 'grefcoco' 
    
    
    # [修改: 提取统一的精度 dtype]
    if args.precision == "fp16":
        dtype = torch.half
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float
        
    for sample_idx, input_dict in enumerate(tqdm.tqdm(val_loader)):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        
        # [修改: 兼容双时相图像的精度转换]
        if "images_t1" in input_dict:
            input_dict["images_t1"] = input_dict["images_t1"].to(dtype)
            input_dict["images_t2"] = input_dict["images_t2"].to(dtype)
            input_dict["images_clip_t1"] = input_dict["images_clip_t1"].to(dtype)
            input_dict["images_clip_t2"] = input_dict["images_clip_t2"].to(dtype)
        else:
            input_dict["images"] = input_dict["images"].to(dtype)
            input_dict["images_clip"] = input_dict["images_clip"].to(dtype)
        
        input_dict["reeval"] = True
        output_dict = model_engine(**input_dict)
        pred_masks = output_dict["pred_masks"][0].ge(0).int()
        gt_masks = output_dict["gt_masks"][0].int()
        
        output_ids = output_dict["output_ids"][0]
        seg_or_rej_index = ((output_ids == args.seg_token_idx) | (output_ids == args.rej_token_idx)).nonzero(as_tuple=True)[0]
        pred_nts = (output_ids[seg_or_rej_index] == args.rej_token_idx)
        assert len(seg_or_rej_index) == len(gt_masks)
        assert len(pred_masks) == len(gt_masks)
            
        for b_idx, (pred, gt) in enumerate(zip(pred_masks, gt_masks)):
            
            if gt.sum() < 1.0: # empty target
                inter_i, union_i, _ = intersectionAndUnionGPU(
                    pred.contiguous().clone(),
                    gt.contiguous().clone(),
                    K=2, ignore_index=255
                )
                inter_i = inter_i.cpu().numpy()
                union_i = union_i.cpu().numpy()
                if pred_nts[b_idx]:
                    nt_tp_meter.update(1.0)
                    g_iou_meter.update(1.0)
                else:
                    nt_fn_meter.update(1.0)
                    g_iou_meter.update(0.0)
                    if is_grefcoco:
                        union_meter.update(union_i)
            else:
                if pred_nts[b_idx]:
                    nt_fp_meter.update(1.0)
                else:
                    nt_tn_meter.update(1.0)
                inter_i, union_i, _ = intersectionAndUnionGPU(
                    pred.contiguous().clone(),
                    gt.contiguous().clone(),
                    K=2, ignore_index=255
                )
                inter_i = inter_i.cpu().numpy()
                union_i = union_i.cpu().numpy()
                this_giou = inter_i / (union_i + 1e-8)
                inter_meter.update(inter_i)
                union_meter.update(union_i)
                g_iou_meter.update(this_giou)

    inter_meter.all_reduce()
    union_meter.all_reduce()
    g_iou_meter.all_reduce()
    nt_tp_meter.all_reduce()
    nt_tn_meter.all_reduce()
    nt_fp_meter.all_reduce()
    nt_fn_meter.all_reduce()
    
    # total_masks = nt_tp_meter.sum + nt_tn_meter.sum + nt_fp_meter.sum + nt_fn_meter.sum
    # masks_have_targets = nt_tn_meter.sum + nt_fp_meter.sum
    N_acc = nt_tp_meter.sum / (nt_tp_meter.sum + nt_fn_meter.sum) # for gt is empty, pred is empty
    T_acc = nt_tn_meter.sum / (nt_tn_meter.sum + nt_fp_meter.sum) # for gt is target, pred is target
    g_iou = g_iou_meter.avg[1]
    c_iou = (inter_meter.sum / (union_meter.sum + 1e-10))[1]
    logger.info(f"[{epoch + 1:d}] {val_loader.dataset.ds} giou: {g_iou:.4f}, ciou: {c_iou:.4f}, N_acc: {N_acc:.4f}, T_acc: {T_acc:.4f}.")
    return g_iou, c_iou, N_acc, T_acc

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape, f"output_shape = {output.shape}, target_shape = {target.shape}"
    output = output.reshape(-1)
    target = target.reshape(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict
