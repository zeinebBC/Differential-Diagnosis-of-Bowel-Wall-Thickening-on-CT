from pathlib import Path
import pandas as pd
import torch.utils.data as data
import torch
import json
import numpy as np
import nibabel as nib
from sklearn.model_selection import StratifiedKFold, train_test_split
from data.utils.cropping import batch_crop_and_save, crop_to_label_region
from data.utils.resampling import batch_resample_and_save
from data.utils.normalizing import process_and_window_dataset
from data.utils.functions_utils import load_volume, pad_to_shape
import os 


class BaseDataset(data.Dataset):
    
    def __init__(
        self,
        patch_size=None,
        dataset_name = None,
        transforms=None,
        num_patches_per_epoch=None,
        path_root=None,
        fold=None,
        split=None,
        return_full_image=False,
        use_labels=None,
        **preprocess_kwargs
        
    ):
        # ----------------------------
        # Paths
        # ----------------------------
        self.path_root = Path(path_root) if path_root is not None else Path("/data/colon_cancer/Classifier") 
        
        self.split = split
        self.return_full_image = return_full_image
        self.dataset_name = dataset_name
        
        self.epoch = 0 
        self.splits_file = self.path_root / self.dataset_name / f"splits.csv" 
        self.labels_file = self.path_root / self.dataset_name / f"labels.csv"
        self.use_labels= use_labels
        # ----------------------------
        # Preprocessing paths
        # ----------------------------

        
        
        if split=="train" or split=="val":
            self.images_path = Path(f"/data/colon_cancer/Classifier/ColonCancer/nnUNetPlans_3d_fullres")
            self.labels_path = Path(f"/data/colon_cancer/Classifier/ColonCancer/nnUNetPlans_3d_fullres")

            #self.images_path = self.path_root / self.dataset_name /"pp_Tr_npz"
            #self.labels_path = self.path_root / self.dataset_name / "resampledTr/labels_resampled"

        elif split=="test":
            self.images_path = self.path_root / self.dataset_name /"pp_Ts_npz"
            self.labels_path = self.path_root / self.dataset_name / "resampledTs/labels_resampled"
        
        # ----------------------------
        # Dataset loading
        # ----------------------------
        if not self.images_path.exists():
            self. preprocess_dataset(preprocess_kwargs)

  


        self.df = pd.read_csv(self.splits_file)
        if fold is not None:
            self.df = self.df[self.df['Fold'] == fold]
        if split is not None:
            self.df = self.df[self.df['Split'] == split]

        self.images = []
        for _, row in self.df.iterrows():
            uid = str(row["UID"])
            target = int(row["target"])
            img_path = Path(self.images_path) / f"{uid}.npz"
            #img_path = Path(self.images_path) / f"{int(uid):03d}.b2nd"
            self.images.append((uid, img_path, target))

        print(f"Loaded {len(self.images)} subjects for Fold={fold}, Split='{split}'")

        self.patch_size = patch_size
        self.transforms = transforms

        self.num_patches_per_epoch = num_patches_per_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch 

    def __len__(self):
        return self.num_patches_per_epoch if self.num_patches_per_epoch else len(self.images)
   
    def __getitem__(self, idx):
    
    
        # ---------- deterministic random setup (optional) ----------
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        patch_seed = 42 + self.epoch * 100000 + worker_id * 1000 + idx
        rng = np.random.default_rng(patch_seed)
        # uid, img_path, target = self.images[rng.integers(0, len(self.images))]

        # For now: pick a random image
        if self.return_full_image:
            uid, img_path, target = self.images[idx]
        else:
            uid, img_path, target = self.images[rng.integers(0, len(self.images))] 


        # ---------- load image ----------
        img = load_volume(img_path)

        # ---------- load label if available ----------
        lbl = None
        if self.use_labels :
            
            lbl_path = self.labels_path / (str(uid) + ".nii.gz")
            #lbl_path = self.labels_path / f"{int(uid):03d}_seg.b2nd"
            if lbl_path.exists():
                lbl = load_volume(lbl_path)
            else:
                print(f" Warning: Label not found for {uid}")
        #################################################################################################################################
        #img, lbl, _ = crop_to_label_region(img[None,...], lbl, spacing= [0.73828125,0.73828125, 0.8999999761581421], margin_min=20 )
        #img = img[0]
        #################################################################################################################################
        # ---------- process full image mode ----------
        if self.return_full_image:
            img_t = torch.from_numpy(img).unsqueeze(0).float()
            if self.transforms:
                img_t = self.transforms(img_t)
            if lbl is not None:
                lbl_t = torch.from_numpy(lbl).unsqueeze(0).float()
                img_t = torch.cat([img_t, lbl_t], dim=0)  # 2 channels
            return {"uid": uid, "source": img_t, "target": torch.tensor(target, dtype=torch.long)}

        # ---------- random crop ----------
        H, W, D = img.shape
        ps = self.patch_size

        def get_crop_coords(dim, patch_dim):
            return rng.integers(0, max(1, dim - patch_dim + 1)) if dim > patch_dim else 0

        x, y, z = [get_crop_coords(s, p) for s, p in zip((H, W, D), ps)]
        img_patch = img[x:x+ps[0], y:y+ps[1], z:z+ps[2]]
        lbl_patch = lbl[x:x+ps[0], y:y+ps[1], z:z+ps[2]] if lbl is not None else None

        
        

        if img_patch.shape != ps:
            img_patch = pad_to_shape(img_patch, ps)
            if lbl_patch is not None:
                lbl_patch = pad_to_shape(lbl_patch, ps)

        # ---------- convert to tensor ----------
        img_t = torch.from_numpy(img_patch).unsqueeze(0).float()
        if self.transforms:
            img_t = self.transforms(img_t)

        if lbl_patch is not None:
            lbl_t = torch.from_numpy(lbl_patch).unsqueeze(0).float()
            img_t = torch.cat([img_t, lbl_t], dim=0)  


        return {
            "uid": uid,
            "source": img_t,
            "target": torch.tensor(target, dtype=torch.long),
            "patch": (x, y, z),
        }
    
    def create_splits(self,n_folds=5, val_fraction=0.1, seed=42, cross_val=False):
        """
        Create stratified train/val splits for cross-validation or a single split.
        Images whose paths contain 'imagesTs' are assigned to the test set and
        excluded from training/validation splitting.
        """
        

        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found at {self.labels_file}. "
                                    "Run generate_target_mapping() first.")

        df = pd.read_csv(self.labels_file)
        print(f"Loaded {len(df)} samples from {self.labels_file}")
        print("Class distribution:", df['target'].value_counts().to_dict())

        # Identify test samples
        test_mask = df['img_path'].str.contains('imagesTs')
        df['Split'] = None
        df.loc[test_mask, 'Split'] = 'test'

        # Only use non-test samples for train/val splits
        trainval_df = df[~test_mask].copy()

        split_col = trainval_df.columns.get_loc('Split')  # Column index for 'Split'

        if cross_val:
            # === Stratified n-fold cross-validation ===
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splits = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(trainval_df, trainval_df['target'])):
                df_fold = trainval_df.copy()
                df_fold['Fold'] = fold

                df_fold.iloc[train_idx, split_col] = 'train'
                df_fold.iloc[val_idx, split_col] = 'val'

                splits.append(df_fold)

            df_splits = pd.concat(splits + [df[test_mask]]).reset_index(drop=True)
            df_splits.to_csv(self.splits_file, index=False)
            print(f"Created {n_folds}-fold train/val splits with test set and saved to {self.splits_file}")

        else:
            # === Single train/val split ===
            train_idx, val_idx = train_test_split(
                range(len(trainval_df)),
                test_size=val_fraction,
                stratify=trainval_df['target'],
                random_state=seed
            )

            trainval_df['Fold'] = 0
            trainval_df.iloc[train_idx, split_col] = 'train'
            trainval_df.iloc[val_idx, split_col] = 'val'

            df_final = pd.concat([trainval_df, df[test_mask]]).reset_index(drop=True)
            df_final.to_csv(self.splits_file, index=False)
            print(f"Created single train/val split with test set and saved to {self.splits_file}")
        

        
    def preprocess_dataset(
        self,
        overwrite_cropping=True,
        overwrite_resample=True,
        overwrite_window=True,
        resample_spacing=(0.7, 0.7, 0.8),
        cross_val=False,
        n_folds=5,
        val_fraction=0.1,
        seed=42,
        path_data= None,
        
    ):
        if self.split.lower() == "test":
            split = "Ts"
        else:
            split = "Tr"

        if path_data is None :
            path_data = self.path_root / self.dataset_name / f"raw_splitted" 


        images_dir = path_data / f"images{split}"
        labels_dir = path_data / f"labels{split}"
     
        # ----------------------------
        # Create splits
        # ----------------------------
        if not self.splits_file.exists():
            self.create_splits(n_folds=n_folds, val_fraction=val_fraction, seed=seed, cross_val=cross_val)

        # ----------------------------
        # Cropping
        # ----------------------------

        path_cropped = self.path_root / self.dataset_name / f"raw_cropped{split}"
        if not path_cropped.exists() or overwrite_cropping:
            batch_crop_and_save(
                images_dir=images_dir,
                labels_dir=labels_dir,
                output_dir=path_cropped,
                margin_min=20.0,
                overwrite=overwrite_cropping,
            )

        # ----------------------------
        # Resampling
        # ----------------------------
        path_resampled = self.path_root / self.dataset_name / f"resampled{split}"
        if not path_resampled.exists() or overwrite_resample:
            batch_resample_and_save(
                root_dir=path_cropped,
                output_dir=path_resampled,
                target_spacing=resample_spacing,
                overwrite=overwrite_resample,
            )

        # ----------------------------
        # Windowing + normalization
        # ----------------------------
        path_prepprocessed =self.path_root / self.dataset_name / f"pp_{split}_npz"
        if not path_prepprocessed.exists() or overwrite_window:
            process_and_window_dataset(
                images_dir=path_resampled / f"images_resampled",
                output_dir=path_prepprocessed,
                window_min=-100,
                window_max=500,
                overwrite=overwrite_window,
            )

        print("Preprocessing complete. Dataset is ready to use.")




    

class ColonCancer(BaseDataset):
  

    def __init__(
        self,
        patch_size=None,
        dataset_name="ColonCancer",
        transforms=None,
        num_patches_per_epoch=None,
        path_root=None,
        fold=None,
        split=None,
        return_full_image=False,
        use_labels=None,
        **preprocess_kwargs
        
    ):
        super().__init__(patch_size,dataset_name, transforms, num_patches_per_epoch, path_root, fold, split,return_full_image,use_labels,**preprocess_kwargs)
        
    
 
        
    def generate_target_mapping(self,images_folder):
        
        DIV_CASES = Path(os.environ.get("CC_DIV_CASES", "/data/colon_cancer/Classifier/filename_mapping.json"))
        images = list(images_folder.glob('*.nii*'))
        
        if not DIV_CASES.exists():
            raise FileNotFoundError(f"Mapping file of diverticulities cases:  {DIV_CASES} not found.")
        # Load known UIDs
        with open(DIV_CASES) as f:
            known_uids = json.load(f)
        known_uids_set = set(known_uids.keys())
        print(f"Loaded {len(known_uids_set)} known UIDs from {DIV_CASES}")
        
        mapping = []
        num_div = 0
        num_cancer = 0
        if Path(self.labels_file).exists():
            existing_df = pd.read_csv(self.labels_file)
            existing_uids = set(existing_df['UID'].astype(str))
            print(f"Loaded {len(existing_uids)} existing UIDs from {self.labels_file}")
        else:
            existing_df = pd.DataFrame()
            existing_uids = set()

        for img_path in images:
            uid = img_path.stem  # filename without extension

            uid = uid.replace('_0000.nii','')
            # Skip if already in existing mapping
            if uid in existing_uids:
                continue
            target = 0 if uid in known_uids_set else 1
            if target == 0:
                num_div +=1
            else:
                num_cancer += 1   
            

            mapping.append({'UID': uid, "img_path" : img_path, 'target': target})

        # Define the correct column order
        columns_order = ['UID', 'img_path', 'target']

        # Create new_df with correct column order
        if mapping:
            new_df = pd.DataFrame(mapping)[columns_order]
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Optional: enforce column order again after concatenation
            final_df = final_df[columns_order]
        else:
            final_df = existing_df[columns_order]

        # Save updated CSV
        final_df.to_csv(self.labels_file, index=False)
        print(f"Saved classification labels for {len(final_df)} images to {self.labels_file}")
        print(f"Number of new diverticulitis cases: {num_div}")
        print(f"Number of new colon cancer cases: {num_cancer}")




    def preprocess_dataset(
        self,
        overwrite_cropping=True,
        overwrite_resample=True,
        overwrite_window=True,
        resample_spacing=(0.7, 0.7, 0.8),
        cross_val=False,
        n_folds=5,
        val_fraction=0.1,
        seed=42,
        path_data= Path(f"/data/colon_cancer/Task101_Colon/raw_splitted/")
    ):
        
      
        if not self.labels_file.exists():
            self.generate_target_mapping(path_data / "imagesTr")
            self.generate_target_mapping(path_data / "imagesTs")


        super().preprocess_dataset( overwrite_cropping=overwrite_cropping,
        overwrite_resample=overwrite_resample,
        overwrite_window=overwrite_window,
        resample_spacing=resample_spacing,
        cross_val=cross_val,
        n_folds=n_folds,
        val_fraction=val_fraction,
        seed=seed,
        path_data= path_data)
