from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNETR
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob

print_config()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = '/cluster/projects/vc/data/mic/open/HNTS-MRG/'

val_images = sorted(glob.glob(os.path.join(data_dir, "test", "*", "preRT", "*T2.nii.gz")))
val_labels = sorted(glob.glob(os.path.join(data_dir, "test", "*", "preRT", "*preRT_mask.nii.gz")))

val_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(val_images, val_labels)]


val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=8)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)


model = UNETR(
    in_channels=1,
    out_channels=3,
    img_size=(96, 96, 32),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    proj_type="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.2,
).to(device)

model.load_state_dict(torch.load("unetr.pth"))
model.eval()
with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        val_inputs, val_labels = (            val_data["image"].to(device),
            val_data["label"].to(device))
        val_outputs = sliding_window_inference(val_inputs, (96, 96, 32), 4, model)
        # plot the slice [:, :, 30]
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image {i}")
        plt.imshow(val_data["image"][0, 0, :, :, 24], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label {i}")
        plt.imshow(val_data["label"][0, 0, :, :, 24])
        plt.subplot(1, 3, 3)
        plt.title(f"output {i}")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 24])
        plt.savefig(f"./results/results{i}.png")