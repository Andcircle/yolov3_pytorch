from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
from utils.parse_config import *
# from test import evaluate
from valid_traffic_sign import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from torch.onnx import OperatorExportTypes


# Only sparse non-short-cut layers
def updateBN(model,s,dontprune):
    for k,m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            if k not in dontprune:
                m.weight.grad.data.add_(s*torch.sign(m.weight.data))

# Shortcut layer need to be handle separately
def dontprune(model):
    non_prunable = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, shortcutLayer):
            # hand code for now
            # -3 sequence
            x = k - 11
            non_prunable.append(x)
            # -1 sequence
            x = k - 3
            non_prunable.append(x)
    return non_prunable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=1, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco_test.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", default="weights/yolov3-tiny.weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--verbose", "-v", default=False, action='store_true', help="Makes the training more verbose")
    parser.add_argument("--logdir", type=str, default="logs", help="Defines the directory where the training log files are stored")
    parser.add_argument("--experiment", type=str, default="yolov3", help="Experiment name")
    parser.add_argument("--sparsity_training", default=True, action='store_true', help="Whether use sparsity training")
    parser.add_argument('--sparse_l1_rate', type=float, default=0.0001, help='Batch norm L1 rate')
    # parser.add_argument("--checkpoints", type=str, default="checkpoints", help="Defines the directory where the checkpoints are saved")
    # parser.add_argument("--onnx", type=str, default="onnx", help="Defines the directory where the onnx file are saved")
    opt = parser.parse_args()
    print(opt)

    logger = Logger(opt.logdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt.checkpoints = os.path.join('experiment', opt.experiment, 'checkpoints')
    opt.onnx = os.path.join('experiment', opt.experiment, 'onnx')
    opt.weights = os.path.join('experiment', opt.experiment, 'weights')

    os.makedirs(opt.checkpoints, exist_ok=True)
    os.makedirs(opt.onnx, exist_ok=True)
    os.makedirs(opt.weights, exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    optimizer = torch.optim.Adam(model.parameters())

    epoch_start=0
    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            # model.load_state_dict(torch.load(opt.pretrained_weights))
            checkpoint = torch.load(opt.pretrained_weights)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch']
            loss = checkpoint['loss']
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, multiscale=opt.multiscale_training, img_size=opt.img_size, transform=AUGMENTATION_TRANSFORMS)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    val_dataset = ListDataset(valid_path, img_size=opt.img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        collate_fn=val_dataset.collate_fn
    )

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    non_prunable = dontprune(model)

    log_graph=True

    for epoch in range(epoch_start, opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()
            if opt.sparsity_training:
                updateBN(model,opt.sparse_l1_rate,non_prunable)

            if log_graph:
            # graph only log once, without loss calc
                logger.graph_summary(model,imgs)
                torch.onnx.export(model, imgs, os.path.join(opt.onnx, "{}.onnx".format(opt.experiment)), verbose=True,
                      operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)
                log_graph=False

            if batches_done % opt.gradient_accumulations == 0:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *["YOLO Layer {}".format(i) for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {to_cpu(loss).item()}"

            # Tensorboard logging
            tensorboard_log = []
            for j, yolo in enumerate(model.yolo_layers):
                for name, metric in yolo.metrics.items():
                    if name != "grid_size":
                        tensorboard_log += [(f"train/{name}_{j+1}", metric)]
            tensorboard_log += [("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            if opt.verbose: print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = evaluate(
                model,
                val_dataloader,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
                class_names=class_names
            )
            
            if metrics_output is not None:
                precision, recall, AP, f1, ap_class, image_detection, image_label= metrics_output
                evaluation_metrics = [
                ("validation/precision", precision.mean()),
                ("validation/recall", recall.mean()),
                ("validation/mAP", AP.mean()),
                ("validation/f1", f1.mean()),
                ]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)
                logger.image_summary('ground_truth_img', image_label, epoch)
                logger.image_summary('prediction_img', image_detection, epoch)

                # Print class APs and mAP
                ap_table = [["Index", "Class name", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                print(AsciiTable(ap_table).table)
                print(f"---- mAP {AP.mean()}")                
            else:
                print( "---- mAP not measured (no detections found by model)")

        if epoch % opt.checkpoint_interval == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, os.path.join(opt.checkpoints, "{}_ckpt_{}.pth".format(opt.experiment, epoch)))
            model.save_darknet_weights(os.path.join(opt.weights, "{}_weights_{}.weights".format(opt.experiment, epoch)))

            if opt.sparsity_training:
                # total = 0
                # for k, m in enumerate(model.modules()):
                #     if isinstance(m, nn.BatchNorm2d):
                #         if k not in non_prunable:
                #             total += m.weight.data.shape[0]
                # bn = torch.zeros(total)
                # index = 0
                # for k, m in enumerate(model.modules()):
                #     if isinstance(m, nn.BatchNorm2d):
                #         if k not in non_prunable:
                #             size = m.weight.data.shape[0]
                #             bn[index:(index + size)] = m.weight.data.abs().clone()
                #             index += size
                # y, i = torch.sort(bn) 
                # number = int(len(y)/5)
                
                bn = np.array([])
                for k, m in enumerate(model.modules()):
                    if isinstance(m, nn.BatchNorm2d):
                        if k not in non_prunable:
                            size = m.weight.data.shape[0]
                            bn = np.append(bn, m.weight.data.clone().abs().cpu().detach().numpy())
                idx = np.argsort(bn)
                bn = bn[idx]
                number = int(len(bn)/5)
        
                print("0~20%%:%f,20~40%%:%f,40~60%%:%f,60~80%%:%f,80~100%%:%f"%(bn[number],bn[2*number],bn[3*number],bn[4*number],bn[-1]))

    torch.onnx.export(model, imgs, os.path.join(opt.onnx, "{}.onnx".format(opt.experiment)), verbose=True,
                      operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)
