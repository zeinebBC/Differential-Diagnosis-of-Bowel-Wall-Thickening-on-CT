from pathlib import Path
from tqdm import tqdm
import torch
import torchio as tio
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
import pandas as pd
from collections import Counter
import json
from types import SimpleNamespace

from classifier.data import basedataset, DataModuleCC
from classifier.models import ResNet
from classifier.scripts.utils.functions import str2bool
from classifier.data.utils.functions_utils import pad_batch_with_channel

# ---------------------------
# Helper: get_model
# ---------------------------
def get_model(cfg):
    if cfg.model.type == "ResNet":
        return ResNet(
            in_ch=cfg.model.in_ch,
            out_ch=cfg.model.out_ch,
            model=cfg.model.resnet_model,
            kwargs_resnet=cfg.model.kwargs_resnet,
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.type}")

def run_pred(model, batch, use_softmax=True):
    source = batch['source']
    if not isinstance(model, ResNet):
        raise ValueError(f"Unknown model type: {type(model)}")
    
    pred = model(source)
    num_classes = pred.shape[1] if pred.ndim > 1 else 1

    if use_softmax:
        pred = torch.sigmoid(pred) if num_classes == 1 else torch.softmax(pred, dim=1)
    
    return pred

# ---------------------------
# Main
# ---------------------------
def main():
    # Load config
    config_file = Path(__file__).resolve().parent.parent / "run_config.json"
    with open(config_file, "r") as f:
        cfg_dict = json.load(f)

    cfg = SimpleNamespace(**cfg_dict)
    cfg.testing = SimpleNamespace(**cfg.testing)
    cfg.model = SimpleNamespace(**cfg.model)

    # Paths
    path_root = Path(os.environ.get("root_path"))
    chkpt_folder = Path(cfg.testing.chkpt_folder)
    path_out = path_root / cfg.testing.output_dir / chkpt_folder.name / cfg.testing.dataset
    path_out.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    # Dataset
    ds_test = basedataset(
        dataset_name=cfg.testing.dataset,
        patch_size=cfg.testing.patch_size,
        split="test",
        return_full_image=True,
        use_labels=cfg.testing.use_labels,
        overwrite_cropping=cfg.testing.overwrite_cropping,
        overwrite_resample=cfg.testing.overwrite_resample,
        overwrite_window=cfg.testing.overwrite_window,
        resample_spacing=tuple(cfg.testing.resample_spacing),
        
    )

    # DataModule
    dm = DataModuleCC(
        ds_test=ds_test,
        batch_size=1,
        pin_memory=True,
        num_workers=cfg.testing.num_workers,
    )

    # Model
    model = get_model(cfg)
    if cfg.testing.use_last:
        model = model.load_last_checkpoint(chkpt_folder)
    else:
        model = model.load_best_checkpoint(chkpt_folder)
    model.to(device)
    model.eval()

    # Inference
    results = []
    aggregation_mode = cfg.testing.aggregation_mode
    patch_size = cfg.testing.patch_size
    patch_overlap = cfg.testing.patch_overlap

    for batch in tqdm(dm.test_dataloader()):
        image, target, uid = batch['source'], batch['target'], batch['uid']
        image, target = image.to(device), target.to(device)
        image = pad_batch_with_channel(image, patch_size, constant_values=0)[0]

        subject = tio.Subject(image=tio.ScalarImage(tensor=image))
        sampler = tio.inference.GridSampler(subject, patch_size=patch_size, patch_overlap=patch_overlap)

        patch_preds_sum = None
        patch_class_votes = []
        n_patches = 0

        for patches_batch in sampler:
            patches = patches_batch['image'][tio.DATA].to(device).unsqueeze(0)
            preds = run_pred(model, {'source': patches}, use_softmax=True)
            preds_cpu = preds.detach().cpu()
            num_classes = preds_cpu.shape[1] if preds_cpu.ndim > 1 else 1

            if aggregation_mode == "average":
                patch_preds_sum = preds_cpu if patch_preds_sum is None else patch_preds_sum + preds_cpu
            elif aggregation_mode == "majority":
                if num_classes == 1:
                    patch_class_votes.extend((preds_cpu > 0.5).int().flatten().tolist())
                else:
                    patch_class_votes.extend(torch.argmax(preds_cpu, dim=1).flatten().tolist())
            else:
                raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")
            n_patches += 1

        # Final aggregation
        if aggregation_mode == "average":
            mean_prob = patch_preds_sum / n_patches
            final_label = int((mean_prob > 0.5).item()) if num_classes == 1 else torch.argmax(mean_prob).item()
        else:
            class_counts = Counter(patch_class_votes)
            final_label = max(class_counts, key=class_counts.get)
            mean_prob = torch.tensor([class_counts[c] / n_patches for c in sorted(class_counts.keys())])

        results.append({'UID': uid[0], 'GT': target.item(), 'NN': final_label, 'NN_pred': mean_prob[0].tolist()})

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(path_out / 'results.csv', index=False)

    # Metrics
    y_true = np.array(df['GT'])
    y_pred = np.array(df['NN'])
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    with open(path_out / "classification_report.txt", "w") as f:
        f.write("=== Classification Report ===\n")
        f.write(report)
        f.write("\n\n=== Confusion Matrix ===\n")
        f.write(np.array2string(cm))


if __name__ == "__main__":
    main()
