import torch 
import torch.nn as nn 
import monai.networks.nets as nets
from models.base_model import BasicClassifier
import torchvision.models as models

import os
def _get_resnet_monai(model):
    return {
        18: nets.resnet18, 34: nets.resnet34, 50: nets.resnet50, 101: nets.resnet101, 152: nets.resnet152
    }.get(model)
    
def _get_resnet_torch(model):
    return {
        18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152
    }.get(model)

class GetLast(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input[-1]

def download_if_missing(url: str, local_path: str):
    """Download a file with wget if it does not exist locally"""
    if not os.path.exists(local_path):
        print(f"Downloading {url} to {local_path} ...")
        os.system(f"wget -O {local_path} {url}")
    return local_path 

class ResNet(BasicClassifier):
    def __init__(self, in_ch, out_ch, spatial_dims=3, model=18, pretrained=False, kwargs_resnet={}, **kwargs):
        emb_ch = kwargs.pop('emb_ch', out_ch)
      
        super().__init__(in_ch, out_ch, spatial_dims, **kwargs)
        
        self.attention_maps = []

        if pretrained:
            if spatial_dims==3:
                resnet = nets.ResNetFeatures(model_name=f'resnet{model}',  spatial_dims=spatial_dims, in_channels=in_ch, pretrained=False)
                url_map = {
                    18: "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18/resolve/main/resnet_18.pth",
                    34: "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet34/resolve/main/resnet_34.pth",
                    50: "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50/resolve/main/resnet_50.pth",
                    101: "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet101/resolve/main/resnet_101.pth",
                    152: "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet152/resolve/main/resnet_152.pth",
                }
                weights_root = f"/data/benchaaben/classifier/models/weights/"
                os.makedirs(weights_root, exist_ok=True)
                weights_path = weights_root + f"resnet{model}_3d.pth"
                download_if_missing(url_map[model], weights_path)
                checkpoint = torch.load(weights_path, map_location="cpu")
                state_dict = checkpoint.get("state_dict", checkpoint)

                # Remove 'module.' prefix if it exists
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v

                resnet.load_state_dict(new_state_dict)


                resnet_out_ch = max([ mod.num_features for name, mod in resnet.layer4[-1]._modules.items() if "bn" in name])
                self.model = nn.Sequential(
                    resnet,
                    GetLast(),
                    nn.AdaptiveAvgPool3d(1),
                    nn.Flatten(1),
                    nn.Linear(resnet_out_ch, emb_ch)
                )
            elif spatial_dims==2:
                Model = _get_resnet_torch(model)
                self.model =  Model(weights=None)
                url_map = {
                    18: "https://download.pytorch.org/models/resnet18-f37072fd.pth",
                    34: "https://download.pytorch.org/models/resnet34-b627a593.pth",
                    50: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
                    101: "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
                    152: "https://download.pytorch.org/models/resnet152-f82ba261.pth",
                }
                weights_root = f"/data/benchaaben/classifier/models/weights/"
                os.makedirs(weights_root, exist_ok=True)
                weights_path = weights_root + f"resnet{model}_2d.pth"
                download_if_missing(url_map[model], weights_path)
                checkpoint = torch.load(weights_path, map_location="cpu")
                state_dict = checkpoint.get("state_dict", checkpoint)

                # Remove 'module.' prefix if it exists
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v

                resnet.load_state_dict(new_state_dict)


                resnet_out_ch = self.model.fc.in_features
                if emb_ch is None:
                    self.model.fc = nn.Identity()
                elif emb_ch != 1000:
                    self.model.fc = nn.Linear(resnet_out_ch, emb_ch)
        else:
            Model = _get_resnet_monai(model)
            self.model = Model(n_input_channels=in_ch, spatial_dims=spatial_dims, num_classes=emb_ch, **kwargs_resnet)
        
   
    def forward(self, source, **kwargs):
        

        output = self.model(source.to(self.device))

        
        return output

    
    

    
    

