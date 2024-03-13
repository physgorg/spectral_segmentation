# train tools for penn dataset

import copy
import os
import time
import torch
from torch import nn

import forge.experiment_tools as fet

def parse_reports(report_dict):
    return {
        k: v.item() if len(v.shape) == 0 else v.detach().clone()
        for k, v in report_dict.items()
    }


def parse_reports_cpu(report_dict):
    return {
        k: v.item() if len(v.shape) == 0 else v.clone().cpu().numpy()
        for k, v in report_dict.items()
    }

def print_reports(report_dict, start_time, epoch, batch_idx, num_epochs, prefix=""):

    reports = ["{}:{:.06f}".format(*item) for item in report_dict.items()]
    report_string = ", ".join(reports)
    if prefix:
        print(prefix, end=": ")

    print(
        "time {:.03f},  epoch: {} [{} / {}]: {}".format(
            time.time() - start_time,
            epoch,
            batch_idx,
            num_epochs,
            report_string,
        )
    )

def log_tensorboard(writer, iteration, report_dict, prefix=""):
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    for k, v in report_dict.items():
        writer.add_scalar(prefix + k, v, iteration)


def log_reports(reports_all, iteration, reports, prefix=""):
    reports["iteration"] = iteration
    if prefix != "":
        for d in reports.keys():
            reports_all[prefix + "_" + d].append(reports[d])
    else:
        for d in reports.keys():
            reports_all[d].append(reports[d])

    return reports_all


def get_checkpoint_iter(checkpoint_iter, checkpoint_dir):
    if checkpoint_iter != -1:
        return checkpoint_iter

    return max(fet.find_model_files(checkpoint_dir).keys())

def load_checkpoint(checkpoint_path, model, opt=None, lr_sched=None):
    print("Restoring checkpoint from '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    # Restore model
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore optimizer
    if opt is not None:
        opt.load_state_dict(checkpoint["model_optimizer_state_dict"])

    # Restore LR schedule
    if lr_sched is not None:
        lr_sched.load_state_dict(checkpoint["model_lr_sched_state_dict"])

    # Update starting epoch
    if checkpoint["epoch"] == "final":
        checkpoint["epoch"] = 0
    start_epoch = checkpoint["epoch"] + 1

    if hasattr(checkpoint, "loss"):
        loss = checkpoint["loss"]
    else:
        loss = None

    return start_epoch, loss

def save_checkpoint(checkpoint_name, epoch, model, opt, data_fold = 0,lr_sched=None, loss=None):
    epoch_ckpt_file = "{}-{}".format(checkpoint_name, epoch)
    print("Saving model training checkpoint to {}".format(epoch_ckpt_file))

    state = {
        "epoch": epoch,
        "data_fold":data_fold,
        "model_state_dict": model.state_dict(),
        "model_optimizer_state_dict": opt.state_dict(),
        "model_lr_sched_state_dict": lr_sched.state_dict()
        if lr_sched is not None
        else None,
    }

    if loss is not None:
        state["loss"] = loss

    torch.save(state, epoch_ckpt_file)
    return epoch_ckpt_file

def delete_checkpoint(checkpoint_name, epoch):
    epoch_ckpt_file = "{}-{}".format(checkpoint_name, epoch)
    os.remove(epoch_ckpt_file)
    print("Deleted checkpoint file at {}".format(epoch_ckpt_file))


def nested_to(x, device, dtype):
    """ Move a list of list of... of tensors to device"""
    try:
        x = x.to(device=device, dtype=dtype)
        return x
    except AttributeError:
        assert isinstance(x, (list, tuple))
        x = type(x)(nested_to(xx, device=device, dtype=dtype) for xx in x)
        return x


def param_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def parameter_analysis(model):
    for n, p in model.named_parameters():
        print(
            "{:>100}   {:>20}Mb   {:>20}".format(
                n, p.numel() * 4 / (1024 ** 2), p.type()
            )
        )
    print(
        "{:>100}   {:>20}Mb".format(
            "total parameters",
            sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2),
        )
    )
