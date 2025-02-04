import datetime as dt
import logging
import os
import time
from typing import Any, Literal

import torch

from src.data_loader import DataContainer
from src.minibatch_manager import MiniBatchManager
from src.model import GraphSAINT
from src.utils import MetricsContainer, calc_f1

logger = logging.getLogger(f"{__name__}")


def model_eval(
    mode: Literal["val", "test"],
    model: torch.nn.Module,
    mb_manager: MiniBatchManager,
    criterion: torch.nn.Module,
    is_sig: bool,
) -> tuple[float, float, float]:
    assert mode in ("val", "test"), f"Invalid mode {mode}"

    nodes = mb_manager.nodes_val if mode == "val" else mb_manager.nodes_train

    model.eval()
    with torch.no_grad():
        mb = mb_manager.get_batch(mode=mode)
        pred, labels = model(mb.adj_norm, mb.nodes)
        loss = (
            criterion(pred, labels)
            * (mb.norm_loss.unsqueeze(1) if is_sig else mb.norm_loss)
        ).sum()
        f1mic, f1mac = calc_f1(labels[nodes], pred[nodes], is_sigmoid=is_sig)
    return loss.item(), f1mic, f1mac


def train(data: DataContainer, **kwds: dict[str, Any]):
    # Initialize minibatch manager
    mb_manager = MiniBatchManager(
        data.adj_full, adj_train=data.adj_train, role=data.role, **kwds
    )

    # Set sampler type
    feats = data.feats if kwds["sampler"] == "fa" else None
    mb_manager.init_sampler(kwds["sampler"], feats=feats)

    # Initialize model
    model = GraphSAINT(feats=data.feats, labels=data.class_arr, **kwds)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f" -- Total number of parameters: {sum(p.numel() for p in model.parameters())} --"
    )
    timestamp = dt.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    model_path = f"torch_models/model_{timestamp}.pkl"

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=kwds["lr"])

    is_sig = kwds["loss"] == "sigmoid"

    # Criterion
    criterion = (
        torch.nn.BCEWithLogitsLoss(reduction="none")
        if is_sig
        else torch.nn.CrossEntropyLoss(reduction="none")
    )

    total_time = 0.0
    best_val_f1mic = -1.0
    for e in range(kwds["n_epochs"]):
        start = time.time()
        batch_c = 0

        # Initialize container for epoch metrics
        train_mc = MetricsContainer()
        val_mc = MetricsContainer()

        while (batch_c * mb_manager.sampler.sg_budget) < mb_manager.nodes_train.shape[
            0
        ]:
            # Reset optimizer and set model to train mode
            model.train()
            optimizer.zero_grad()

            # Retrieve mini batch from the manager
            mb = mb_manager.get_batch(mode="train")

            # Forward pass
            pred, labels = model(mb.adj_norm, mb.nodes)

            # (Normalized) Loss computation
            unorm_loss = criterion(pred, labels)
            loss = (
                unorm_loss * (mb.norm_loss.unsqueeze(1) if is_sig else mb.norm_loss)
            ).sum()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            # Save metrics to containers
            if batch_c % kwds["eval_train_every"] == 0:
                mic, mac = calc_f1(labels, pred, is_sigmoid=is_sig)
                train_mc.append(loss=loss.item(), f1mic=mic, f1mac=mac)

            batch_c += 1

        time_epoch = time.time() - start
        total_time += time_epoch

        logger.info(f"Epoch time: {time_epoch:.4f} sec.")

        # Compute metrics on validation set
        if (e + 1) % kwds["eval_val_every"] == 0:
            loss, f1mic, f1mac = model_eval(
                mode="val",
                model=model,
                mb_manager=mb_manager,
                criterion=criterion,
                is_sig=is_sig,
            )
            val_mc.append(loss=loss, f1mic=f1mic, f1mac=f1mac)

            # Save model in case validation acc. is best ever
            if f1mic > best_val_f1mic:
                best_val_f1mic = f1mic
                if not os.path.exists(os.path.dirname(os.path.abspath(model_path))):
                    os.makedirs(os.path.dirname(os.path.abspath(model_path)))

                logger.info("Saving model to file ...")

                torch.save(model.state_dict(), model_path)

        # Log metrics
        logger.info(
            f"TRAIN (epoch avg): loss = {train_mc.loss:.4f}\tmic = {train_mc.f1mic:.4f}\t mac = {train_mc.f1mac:.4f}"
        )
        logger.info(
            f"VAL (for epoch {e + 1}): loss = {val_mc.loss:.4f}\tmic = {val_mc.f1mic:.4f}\t mac = {val_mc.f1mac:.4f}\n"
        )

    # Load best model (if needed)
    if best_val_f1mic >= 0:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        logger.info("Restoring model ...")

    # Compute test acc.
    loss, f1mic, f1mac = model_eval(
        mode="test",
        model=model,
        mb_manager=mb_manager,
        criterion=criterion,
        is_sig=is_sig,
    )

    logger.info(f"TEST: loss = {loss:.4f}\tmic = {f1mic:.4f}\tmac = {f1mac:.4f}\n")
    logger.info(f"Total training time: {total_time:.4f} sec.")
    logger.info(f"Average time per epoch: {total_time/kwds["n_epochs"]:.4f} sec.")
