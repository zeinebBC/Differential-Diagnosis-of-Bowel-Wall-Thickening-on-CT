
from pathlib import Path
from tqdm import tqdm
import torch 
import torchio as tio
import numpy as np 
from sklearn.metrics import  confusion_matrix, classification_report
import os
import pandas as pd 
from classifier.data import ColonCancer, BaseDataset 
import torchio as tio
from collections import Counter
from classifier.data import DataModuleCC
from classifier.models import ResNet
from classifier.scripts.utils.functions import str2bool
from classifier.data.utils.functions_utils import pad_batch_with_channel
import json 

def get_model(config):
    if config["model"]["type"] == 'ResNet':

        return ResNet(
            in_ch=config["model"]["in_ch"],
            out_ch=config["model"]["out_ch"],
            model=config["model"].get("resnet_model", 18),
            kwargs_resnet=config["model"].get("kwargs_resnet", {}),
        )
    else:
        raise ValueError(f"Unknown model: {config['model']['type']}")


def run_pred(model, batch, use_softmax=True):
    source = batch['source']

    # Optionally check for supported models
    if not isinstance(model, ResNet):
        raise ValueError(f"Unknown model type: {type(model)}")

    # Forward pass
    pred = model(source)
    num_classes = pred.shape[1] if pred.ndim > 1 else 1  # detect binary case

    if use_softmax:
        if num_classes == 1:
            # Binary classification: output single prob via sigmoid
            pred = torch.sigmoid(pred)
        else:
            # Multiclass: use softmax along class dim
            pred = torch.softmax(pred, dim=1)

    return pred
     






def main():
    

    """
    parser.add_argument('--model_name', type=str, required=True, choices=['ResNet'])
    parser.add_argument('--chkpt_folder', default='/data/benchaaben/classifier/runs_Resnet/ColonCancer/ResNet_2025_10_15_094431', type=str)
    parser.add_argument('--output_dir', default='/data/benchaaben/classifier/runs_Resnet/inf_output', type=str)
    parser.add_argument("--seg_path", type=str, help="Path to segmentation labels", default='/data/colon_cancer/Classifier/resampledTs/labels_resampled')
    parser.add_argument("--transforms", type=str, default=None)
    parser.add_argument('--patch_size', nargs=3, type=int, default=[64, 256, 256])
    parser.add_argument('--resample_spacing', nargs=3, type=float, default=[1, 1, 1])
    parser.add_argument('--patch_overlap', nargs=3, type=int, default=[32, 128, 128], help="Overlap for grid sampler")
    parser.add_argument('--aggregation_mode', type=str, default='average', choices=['average', 'majority'], help="Aggregation mode for patch predictions")
    parser.add_argument('--path_root', type=str, default='/data/colon_cancer')
    parser.add_argument('--overwrite_cropping', type=str2bool, default=True)
    parser.add_argument('--overwrite_resample', type=str2bool, default=True)
    parser.add_argument('--overwrite_window', type=str2bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_last', type=str2bool, default=False, help="Whether to use the last checkpoint instead of the best one")
    """
    config_file =Path(__file__).resolve().parent.parent / "run_config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)

    
    path_root = Path(os.environ.get("root_path"))
    chkpt_folder = config["testing"].get("chkpt_folder", "")
    output_dir = path_root / config["testing"].get("output_dir", "")
    use_labels = config["testing"].get("use_labels", True)
    transforms = config["testing"].get("transforms", None)
    patch_size = config["testing"].get("patch_size", [64, 256, 256])
    resample_spacing = config["testing"].get("resample_spacing", [1, 1, 1])
    patch_overlap = config["testing"].get("patch_overlap", [32, 128, 128])
    aggregation_mode = config["testing"].get("aggregation_mode", "average")
   
    overwrite_cropping = config["testing"].get("overwrite_cropping", True)
    overwrite_resample = config["testing"].get("overwrite_resample", True)
    overwrite_window = config["testing"].get("overwrite_window", True)
    num_workers = config["testing"].get("num_workers", 8)
    use_last = config["testing"].get("use_last", False)
    dataset_name = config["testing"].get("dataset", False)

    # --- Setup output folder ---
    path_out = path_root / Path(output_dir) / Path(chkpt_folder).name
    path_out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')

    if dataset_name == "Dataset100_CC":
          
        ds_test = ColonCancer(
            dataset_name= dataset_name,
            patch_size=patch_size,
            transforms=transforms,
            split="test",
            return_full_image=True,
            use_labels=use_labels,
            overwrite_cropping=overwrite_cropping,
            overwrite_resample=overwrite_resample,
            overwrite_window=overwrite_window,
            resample_spacing=tuple(resample_spacing),
        )

      
    else:
        ds_test = BaseDataset(
            dataset_name= dataset_name,
            patch_size=patch_size,
            transforms=transforms,
            split="test",
            return_full_image=True,
            use_labels=use_labels,
            overwrite_cropping=overwrite_cropping,
            overwrite_resample=overwrite_resample,
            overwrite_window=overwrite_window,
            resample_spacing=tuple(resample_spacing),)

       
      


    dm = DataModuleCC(
        ds_test=ds_test,
        batch_size=1,
        pin_memory=True,
        num_workers=num_workers,
    )

    # --- Initialize model ---
    model = get_model(config)
    if use_last:
        model = model.load_last_checkpoint(chkpt_folder)
    else:
        model = model.load_best_checkpoint(chkpt_folder)
    model.to(device)
    model.eval()

    # --- Inference ---
    results = []
    for n, batch in enumerate(tqdm(dm.test_dataloader())):
        image, target, uid = batch['source'], batch['target'], batch['uid']
        image, target = image.to(device), target.to(device)
        image = pad_batch_with_channel(image, patch_size, constant_values=0)[0]

        subject = tio.Subject(image=tio.ScalarImage(tensor=image))
        sampler = tio.inference.GridSampler(subject, patch_size=patch_size, patch_overlap=patch_overlap)

        patch_preds_sum = None
        patch_class_votes = []
        n_patches = 0

        for patches_batch in sampler:
            patches = patches_batch['image'][tio.DATA].to(device)
            patches = patches.unsqueeze(0)
            #patches = patches.permute(0, 3, 2, 1).unsqueeze(0)
            preds = run_pred(model, {'source': patches}, use_softmax=True)
            preds_cpu = preds.detach().cpu()
            num_classes = preds_cpu.shape[1] if preds_cpu.ndim > 1 else 1

            if aggregation_mode == "average":
                patch_preds_sum = preds_cpu if patch_preds_sum is None else patch_preds_sum + preds_cpu
            elif aggregation_mode == "majority":
                if num_classes == 1:
                    patch_classes = (preds_cpu > 0.5).int().flatten().tolist()
                else:
                    patch_classes = torch.argmax(preds_cpu, dim=1).flatten().tolist()
                patch_class_votes.extend(patch_classes)
            else:
                raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")
            n_patches += 1

        # --- Final aggregation ---
        if aggregation_mode == "average":
            mean_prob = patch_preds_sum / n_patches
            if num_classes == 1:
                final_label = int((mean_prob > 0.5).item())
                mean_prob = mean_prob.flatten()
            else:
                final_label = torch.argmax(mean_prob).item()
        elif aggregation_mode == "majority":
            class_counts = Counter(patch_class_votes)
            final_label = max(class_counts, key=class_counts.get)
            mean_prob = torch.tensor([class_counts[c] / n_patches for c in sorted(class_counts.keys())])

        results.append({'UID': uid[0], 'GT': target.item(), 'NN': final_label, 'NN_pred': mean_prob[0].tolist()})

    print(f"Finished inference using {aggregation_mode.upper()} aggregation mode")

    # --- Save results ---
    df = pd.DataFrame(results)
    df.to_csv(path_out / 'results.csv', index=False)

    y_true = np.array(df['GT'])
    y_pred = np.array(df['NN'])

    # --- Confusion matrix and report ---
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
   


    with open(path_out / "classification_report.txt", "w") as f:
        f.write("=== Classification Report ===\n")
        f.write(report)
        f.write("\n\n=== Confusion Matrix ===\n")
        f.write(np.array2string(cm))

if __name__ == "__main__":
    main()