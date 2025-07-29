import pathlib
import rasterio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from collections import defaultdict
from tqdm import tqdm
import torch
from pathlib import Path
from torch import nn
from typing import Dict
from torchvision import models, transforms
from PIL import Image


def get_file_paths(folders: Dict[str, pathlib.Path]) -> Dict[str, List[pathlib.Path]]:
    """
    Collects TIFF file paths for each variable folder.
    
    Returns:
        dict: Mapping from variable names to sorted lists of Paths.
    """
    paths = {}
    for var, folder in folders.items():
        paths[var] = sorted(folder.glob('*.tif*'))  # matches both .tif and .tiff
    return paths


def construct_filename(var: str, scene_id: Union[str, int]) -> str:
    """
    Constructs the appropriate filename for a given variable and scene_id.
    """
    sid = str(scene_id)
    if var == 'wet':
        return f"treatment_scene_{sid}.tiff"
    elif var == 'outcome_pre':
        return f"outcome_scene_{sid}.tiff"
    elif var == 'outcome_post':
        return f"outcome_post_scene_{sid}_gaussian.tiff"
    elif var == 'theta':
        return f"Theta_scene_{sid}.tiff"
    elif var == 'theta_directXout':
        return f"Theta_DirectXOutgoing_scene_{sid}.tiff"
    elif var == 'ite_direct':
        return f"ITE_direct_scene_{sid}.tiff"
    elif var == 'ITE_Outgoing':
        return f"ITE_Outgoing_scene_{sid}.tiff"
    elif var == 'ite_indirect':
        return f"ITE_indirect_scene_{sid}.tiff"
    elif var == 'ite_total':
        return f"ITE_total_scene_{sid}.tiff"
    elif var == 'theta_outgoing':
        return f"Theta_Outgoing_scene_{sid}.tiff"
    elif var == 'dem':
        return f"DEM_{sid}.tiff"
    elif var == 'cap':
        return f"CAPITAL_1996_{sid}.tiff"
    else:
        raise ValueError(f"Unknown variable type: {var}")


def compute_tabular_features(
    folders: Dict[str, pathlib.Path],
    scene_ids: List[Union[str, int]],
    output_dir: Optional[pathlib.Path] = None,
) -> pd.DataFrame:
    """
    Compute mean and standard deviation for each variable raster per scene.

    Parameters
    ----------
    folders : dict
        Mapping of variable name to folder Path.
    scene_ids : list of str or int
        Scene identifiers corresponding to file suffixes.
    output_dir : Path, optional
        Directory where 'features.csv' will be saved.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per scene_id and columns '<var>_mean' and '<var>_std'.
    """
    records = defaultdict(dict)

    for scene in scene_ids:
        scene_str = str(scene)
        for var, folder in folders.items():
            filename = construct_filename(var, scene_str)
            fp = folder / filename
            if not fp.exists():
                print(f"Warning: missing file for var='{var}', scene='{scene_str}': {fp}")
                continue
            try:
                with rasterio.open(fp) as src:
                    arr = src.read(1).astype(float)
                    records[scene_str][f'{var}_mean'] = np.nanmean(arr)
                    records[scene_str][f'{var}_std'] = np.nanstd(arr)
            except Exception as e:
                print(f"Error reading {fp}: {e}")

    df = pd.DataFrame.from_dict(records, orient='index') \
                     .reset_index() \
                     .rename(columns={'index': 'scene_id'})

    # Save
    out_dir = pathlib.Path(output_dir) if output_dir else pathlib.Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'features.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved features to: {out_path}")

    return df

def build_embedding_model(
    device: Optional[torch.device] = None,
    model_type: str = 'resnet50',
    conv_layer: int = 4
) -> nn.Module:
    """
    Loads a pretrained model and retains feature maps up to the specified ResNet block.
    
    Parameters
    ----------
    device : torch.device, optional
        Device to load model onto (CPU/GPU).
    model_type : str
        One of 'resnet50', 'resnet18', or 'efficientnet_b0'.
    conv_layer : int
        Which ResNet block to include (1–4). Defaults to final layer (4).
    
    Returns
    -------
    nn.Module
        A feature extractor ending with AdaptiveAvgPool and Flatten.
    """
    if model_type == 'resnet50':
        base = models.resnet50(pretrained=True)
    elif model_type == 'resnet18':
        base = models.resnet18(pretrained=True)
    elif model_type == 'efficientnet_b0':
        base = models.efficientnet_b0(pretrained=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # For ResNet: initial layers + selectable blocks
    modules = [base.conv1, base.bn1, base.relu, base.maxpool]
    if conv_layer >= 1: modules.append(base.layer1)
    if conv_layer >= 2: modules.append(base.layer2)
    if conv_layer >= 3: modules.append(base.layer3)
    if conv_layer >= 4: modules.append(base.layer4)
    modules.extend([nn.AdaptiveAvgPool2d((1,1)), nn.Flatten()])

    model = nn.Sequential(*modules)
    model.eval()
    if device:
        model.to(device)
    return model

from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import rasterio

def compute_image_embeddings(
    folders: Dict[str, Path],
    scene_ids: List[Union[str, int]],
    var: str,
    model: nn.Module,
    device: Optional[torch.device] = None,
    img_size: int = 256,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Computes embeddings for each scene TIFF in the specified variable folder.

    Uses `construct_filename()` to determine filenames.
    Converts binary raster (0/1) to 0–255 grayscale and repeats to RGB.

    Returns
    -------
    pd.DataFrame with one row per scene_id and embedding columns.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        ),
    ])

    folder = folders[var]
    records = []
    
    for scene in tqdm(scene_ids, desc=f'Embedding {var} scenes'):
        scene_str = str(scene)
        try:
            filename = construct_filename(var, scene_str)
        except ValueError as e:
            print(f"Skipping unknown var: {var} — {e}")
            continue
        fp = folder / filename
        if not fp.exists():
            print(f"Missing file for var='{var}', scene='{scene_str}': {fp}")
            continue

        try:
            with rasterio.open(fp) as src:
                arr = src.read().astype('float32')

                # Convert 1s to 255s for better contrast (simulate grayscale)
                arr *= 255.0

                # Repeat grayscale to RGB if needed
                if arr.shape[0] == 1:
                    arr = np.repeat(arr, 3, axis=0)

            img = Image.fromarray(np.moveaxis(arr, 0, -1).astype('uint8'))
            inp = transform(img).unsqueeze(0)
            if device:
                inp = inp.to(device)
            with torch.no_grad():
                emb = model(inp).cpu().numpy().squeeze()

        except Exception as e:
            print(f"Failed to process {fp}: {e}")
            continue

        rec = {'scene_id': scene_str}
        for i, val in enumerate(emb):
            rec[f'emb_{i}'] = val
        records.append(rec)

    df = pd.DataFrame(records)

    out_dir = Path(output_dir) if output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{var}_embeddings.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved embeddings to: {out_path}")

    return df

def compute_multi_embeddings(
    folders: Dict[str, Path],
    scene_ids: List[Union[str,int]],
    vars:     List[str],     # e.g. ['wet','dem','cap']
    model:    nn.Module,
    device:   Optional[torch.device]=None,
    img_size: int    = 256,
    output_dir: Optional[Path]=None
) -> pd.DataFrame:
    """
    Compute a 3-channel embedding from the stack [wetland, DEM, Capital].
    """

    # transforms for the 3-channel PIL image
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],  # ImageNet means
            std =[0.229,0.224,0.225]   # ImageNet stds
        ),
    ])

    records = []
    for scene in tqdm(scene_ids, desc="Embedding stacked scenes"):
        mats = []
        for var in vars:
            fp = folders[var] / construct_filename(var, str(scene))
            if not fp.exists():
                print(f"Missing {var} for scene {scene}")
                break

            # read & resize
            with rasterio.open(fp) as src:
                arr = src.read(1).astype('float32')
            # resample to img_size using PIL:
            im = Image.fromarray(arr)
            im = im.resize((img_size,img_size), resample=Image.BILINEAR)
            arr_r = np.array(im, dtype=np.float32)

            # scale to [0,255]:
            if var=='wet':
                arr_r = arr_r * 255.0              # binary→255
            else:
                # DEM/CAP: min-max normalize per scene
                mn, mx = arr_r.min(), arr_r.max()
                arr_r = 255 * (arr_r - mn) / (mx - mn)

            mats.append(arr_r)

        else:
            # only if no break
            stacked = np.stack(mats, axis=0).astype('uint8')  # (3,H,W)
            img      = Image.fromarray(np.moveaxis(stacked,0,-1))  # (H,W,3)
            inp      = transform(img).unsqueeze(0)
            if device:
                inp = inp.to(device)
            with torch.no_grad():
                emb = model(inp).cpu().numpy().squeeze()

            rec = {'scene_id': str(scene)}
            rec.update({f'emb_{i}':float(v) for i,v in enumerate(emb)})
            records.append(rec)

    df = pd.DataFrame(records)
    out_dir = Path(output_dir) if output_dir else Path.cwd()
    out_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_dir/f"stacked_embeddings.csv", index=False)
    return df


def combine_features(
    tab_df: pd.DataFrame,
    **pca_dfs: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge tabular features with multiple PCA-DataFrames on scene_id.
    
    Parameters
    ----------
    tab_df : pd.DataFrame
        Your base table (must have 'scene_id').
    **pca_dfs : pd.DataFrame
        Named PCA tables, e.g. pca_wet, pca_dem, etc., each with
        columns ['scene_id','PC1',...,'PCn'].
    
    Returns
    -------
    pd.DataFrame
        A single DataFrame with all features + prefixed PCs.
    """
    combined = tab_df.copy()
    for key, pca in pca_dfs.items():
        if not key.startswith('pca_'):
            raise ValueError(f"Expected PCA args to start with 'pca_'; got {key}")
        var = key[len('pca_'):]
        # rename PC columns to var_PC1, var_PC2, ...
        rename_map = {
            col: f"{var}_{col}"
            for col in pca.columns
            if col != 'scene_id'
        }
        pca_renamed = pca.rename(columns=rename_map)
        combined = combined.merge(pca_renamed, on='scene_id', how='left')
    return combined


