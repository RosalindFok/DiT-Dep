import re
import ants
import time
import shutil
import random
import numpy as np
import pandas as pd 
import nibabel as nib 
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from sklearn.model_selection import KFold
from nilearn import maskers, connectome, image

from utils import write_json, read_json
from config import seed, n_splits, Group, Experiment_Config, Mild_Config
from path import Raw_Data_Dir_Path, Running_File_Dir_Path, Brain_Atlas_Dir_Path

random.seed(seed)

@dataclass(frozen=True)
class Gender:
    Unspecified : int = 0
    Male : int = 1
    Female : int = 2
@dataclass(frozen=True)
class Hand:
    X : int = 0
    R : int = 1
    L : int = 2

# get the orientation of MRI
get_orientation = lambda path : "".join(nib.aff2axcodes(nib.load(path).affine)) # input : path, output : str

def get_yeo_network_of_brainnetome() -> None:
    # functional networks
    if not Brain_Atlas_Dir_Path.network_json_path.exists():
        # subregion_func_network_Yeo_updated
        pd_frame = pd.read_csv(Brain_Atlas_Dir_Path.network_csv_path, sep=",")
        mapped_dict = defaultdict(lambda: defaultdict(list))
        labels_networks = pd_frame[["Label", "Yeo_7network", "Yeo_17network"]]
        for (name, yeo_frame) in [
            ("Yeo_7network" , pd_frame.iloc[ 2:9 , 10:12]),
            ("Yeo_17network", pd_frame.iloc[12:29, 10:12])
        ]:
            for _, row in yeo_frame.iterrows():
                # label-1: 0-based indices in functional connectivity matrix
                mapped_dict[name][row.iloc[1]] = [x-1 for x in labels_networks.loc[labels_networks[name]==int(row.iloc[0]), "Label"].tolist()]
        write_json(json_path=Brain_Atlas_Dir_Path.network_json_path, dict_data=mapped_dict)

def register(fixed_path: str, moving_path: str, outprefix: str = "") -> tuple[ants.ANTsImage, list[str]]:
    """Registers a moving image to a fixed image using ANTsPy SyN algorithm.

    This function performs deformable registration of a moving image to a fixed image using ANTsPy's
    SyN (Symmetric Normalization) transformation, with mutual information (MI) as the similarity metric.
    It returns the registered (warped) image and the list of files containing the forward (moving-to-fixed)
    transformation. If `outprefix` is not specified, the registration output files will be written to the
    system's temporary directory.

    Args:
        fixed_path (str): Path to the fixed/reference image file (NIfTI or other ANTs-compatible format).
        moving_path (str): Path to the moving/source image file to be registered.
        outprefix (str, optional): Prefix for naming the output files produced by ANTsPy registration.
            If not provided, files will be saved in the system's temp directory with a default prefix.

    Returns:
        tuple[ants.ANTsImage, list[str]]: 
            - The registered (warped) moving image as an ANTsImage object.
            - List of strings containing the file paths to the forward transformation(s) created.

    Side Effects:
        - Writes transformation files and possibly intermediate result files to disk, location depending
          on `outprefix` (defaults to system temp directory).
        - Reads images from disk.

    """
    # Note: If `outprefix` is '', ANTsPy will use the system's temp directory for output files.
    
    # Load the fixed/reference image from disk.
    fixed_image = ants.image_read(fixed_path)
    
    # Load the moving/source image from disk.
    moving_image = ants.image_read(moving_path)
    
    # Resample the moving image to have the same shape and spacing as the fixed image.
    # `1` specifies isotropic spacing, `0` means linear interpolation.
    moving_image = ants.resample_image(moving_image, fixed_image.shape, 1, 0)
    
    # Perform symmetric normalization (SyN) registration using the MI metric.
    # Returns a dict with the warped image and the transformation filenames.
    registered_image = ants.registration(
        fixed=fixed_image,
        outprefix=outprefix,
        moving=moving_image,
        type_of_transform="SyN",
        aff_metric="MI"
    )
    
    # Return the registered image and the forward transform list.
    # "warpedmovout" is the warped moving image.
    # "fwdtransforms" is a list of forward transformation file paths.
    return registered_image["warpedmovout"], registered_image["fwdtransforms"]

def split_4dantsImageFrame_into_3d(image_4d: ants.ANTsImage) -> str:
    """Converts a singleton 4D ANTsImage to a 3D image and saves it as a temporary NIfTI file.
    
    This function takes an ANTsImage object that has four dimensions, with the last dimension being 1 (i.e., a singleton 
    fourth dimension), and removes that dimension so the image becomes a true 3D image. Due to known bugs in ANTsPy's 
    dimensionality handling, the function writes the image to disk and employs nibabel to safely remove the extra 
    dimension, overwriting the temporary file.
    
    Args:
        image_4d (ants.ANTsImage): Input image with four dimensions, where the fourth (time/frame) dimension is 1.
    
    Returns:
        str: Path to the temporary 3D NIfTI file created after removing the singleton dimension.
    
    Raises:
        AssertionError: If the input image does not have four dimensions or its last dimension is not 1.
    
    Side Effects:
        Writes and overwrites a file named 'temp.nii' in the current working directory.
    """
    # Ensure the input image is 4D with the last dimension equal to 1; otherwise raise an error.
    assert image_4d.shape[-1] == 1 and image_4d.dimension == 4, (
        f"image_4d.shape={image_4d.shape}, image_4d.dimension={image_4d.dimension}"
    )
    # Specify the temporary file name for writing the intermediate 4D image.
    temp_path = "temp.nii"
    
    # Write the 4D ANTsImage to disk as a NIfTI file using ANTsPy's image_write.
    ants.image_write(image_4d, temp_path)
    
    # Due to bugs in ANTsPy's way of dropping dimensions,
    # reload the NIfTI file using nibabel and safely squeeze the singleton dimension.
    img = nib.load(temp_path)
    # Squeeze the last (singleton) dimension, creating a 3D data array.
    new_img = nib.Nifti1Image(
        img.get_fdata().squeeze(-1),  # Remove the last dimension explicitly.
        img.affine,
        img.header
    )
    # Overwrite the temp file with the new 3D NIfTI image.
    nib.save(new_img, temp_path)
    
    # Return the path to the resulting 3D image file.
    return temp_path

def adjust_dim_of_antsImage(image_5d_path : Path) -> None:
    """Adjusts the dimension of a 5D NIfTI image to 4D by removing the singleton dimension.  

    This function loads a 5D NIfTI image from the specified path, checks if the image's dimension is 5,  
    and if so, removes the singleton dimension to convert it to a 4D image. The updated image is saved,  
    overwriting the original file. The function prints the elapsed time required for this operation.  

    Args:  
        image_5d_path (Path): Path object pointing to the 5D NIfTI image file.  

    Returns:  
        None: This function does not return any value. The input file is updated in place.  

    Side Effects:  
        Overwrites the original image file at image_5d_path with the adjusted 4D image.  
        Prints the time taken to complete the dimension adjustment.  
    """  
    start_time = time.time()
    image_5d = nib.load(image_5d_path)
    if image_5d.header['dim'][0] == 5:
        new_data = image_5d.get_fdata().squeeze()
        new_header = image_5d.header
        new_header.set_data_shape(new_data.shape)
        new_img = nib.Nifti1Image(new_data, image_5d.affine, image_5d.header)
        nib.save(new_img, image_5d_path)
        end_time = time.time()
        print(f"It took {end_time - start_time:.2f} seconds to adjust dim of {image_5d_path}")

def extract_motion_parameters(transform_matrix : ants.ANTsTransform) -> list[float, float, float, float, float, float]:  
    """Extracts 6 rigid-body motion parameters from an ANTs transformation matrix.  
    
    This function decomposes an ANTs transformation matrix into 6 motion parameters,  
    consisting of 3 rotation angles (in degrees) and 3 translation values. The rotation  
    parameters are extracted using the principles of rotation matrix decomposition into  
    Euler angles (pitch, roll, yaw).  
    
    Args:  
        transform_matrix: An ANTs transformation matrix object (ANTsTransform)   
            containing the affine transformation parameters. This should be a  
            rigid or affine transformation.  
    
    Returns:  
        A list of 6 float values representing the motion parameters in the  
        following order: [rx, ry, rz, tx, ty, tz], where:  
            - rx: Rotation around X-axis in degrees (pitch)  
            - ry: Rotation around Y-axis in degrees (roll)  
            - rz: Rotation around Z-axis in degrees (yaw)  
            - tx: Translation along X-axis in input space units  
            - ty: Translation along Y-axis in input space units  
            - tz: Translation along Z-axis in input space units  
    
    Notes:  
        - The implementation assumes a particular order in the transform_matrix.parameters  
          array where the rotation components are stored in the first 9 elements and  
          the translation components in the last 3 elements.  
        - The conversion from radians to degrees is performed using the factor 180.0/π.  
    """ 
    # Extract parameters from transformation matrix  
    params = transform_matrix.parameters  
    
    # Extract translation parameters (last 3 elements)  
    tx, ty, tz = params[-3:]  
    
    # Extract rotation matrix and convert to Euler angles  
    rx = np.arctan2(params[7], params[8]) * 180.0 / np.pi  
    ry = np.arcsin(-params[6]) * 180.0 / np.pi  
    rz = np.arctan2(params[3], params[0]) * 180.0 / np.pi  
    
    return [rx, ry, rz, tx, ty, tz]  

def head_motion_correction(func4d_path: Path) -> ants.ANTsImage:  
    """  
    Perform head motion correction on a 4D functional image.  

    Args:  
        func4d_path: Path to the 4D functional image file.  

    Returns:  
        A 4D ANTsImage with corrected volumes after motion correction.  
    """  
    func_image = ants.image_read(str(func4d_path))  
    
    # Get the number of time points (volumes)  
    n_volumes = func_image.shape[-1]  
    
    # Extract the reference volume  
    reference_volume = 0  
    reference = ants.slice_image(func_image, axis=3, idx=reference_volume)  
    
    # Initialize storage for motion parameters  
    motion_params = []  
    
    # Initialize storage for corrected volumes  
    corrected_volumes = []  
    corrected_volumes.append(reference)  # The first volume (reference) remains unchanged  
    
    # Perform registration for each volume  
    for i in tqdm(range(n_volumes), desc=f"Head motion correcting {func4d_path.stem.split('_')[0]}", leave=True):  
        if i == reference_volume:  
            # Skip the reference volume  
            motion_params.append(np.zeros(6))  # Add zero motion parameters  
            continue  
        
        # Extract the current volume  
        moving = ants.slice_image(func_image, axis=3, idx=i)  
        
        # Perform rigid-body registration  
        registration = ants.registration(  
            fixed=reference,  
            moving=moving,  
            type_of_transform='Rigid',  
            verbose=False  
        )  
        
        # Store the registered volume  
        corrected_volumes.append(registration['warpedmovout'])  
        
        # Extract the transformation matrix  
        transform = registration['fwdtransforms'][0]  
        matrix = ants.read_transform(transform)  
        
        # Extract 6 motion parameters (3 rotations and 3 translations) from the ANTs transform  
        params = extract_motion_parameters(matrix)  
        motion_params.append(params)  
    
    # Merge all corrected volumes into a 4D image  
    corrected_4d = ants.merge_channels(corrected_volumes)  
    return corrected_4d

def plot_time_series(time_series : np.ndarray, saved_dir_path : Path, region_names : dict[int, str] = None, dpi : int = 300, img_format : str = "png") -> None:
    n_timepoints, n_regions = time_series.shape

    if region_names is None:
        region_names_list = [f"Region {i+1}" for i in range(n_regions)]
    elif isinstance(region_names, dict):
        region_names_list = [region_names[i] for i in range(n_regions)]
    else:
        region_names_list = list(region_names)
    assert len(region_names_list) == n_regions, f"len(region_names) != n_regions, {len(region_names_list)} != {n_regions}"

    time_series_dir = Path(saved_dir_path) / "time_series"
    time_series_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(n_regions), desc="Plotting time series", leave=True):
        fig, ax = plt.subplots(figsize=(10, 1.5))
        ax.plot(time_series[:, i], color="black", linewidth=0.5)
        ax.set_yticks([])
        ax.set_ylabel(region_names_list[i], rotation=0, labelpad=20, fontsize=8, va="center")
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.set_xlabel("Timepoints")
        plt.tight_layout()

        region_name_safe = re.sub(r'[\\/:*?"<>| ]', "_", region_names_list[i])
        filename = f"{i+1}_{region_name_safe}.{img_format}"
        filepath = time_series_dir / filename

        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', format=img_format)
        plt.close(fig)
    # compact
    archive_path = shutil.make_archive(str(time_series_dir), "zip", root_dir=str(time_series_dir))
    if Path(archive_path).exists():
        shutil.rmtree(time_series_dir)

def plot_fc_matrix(fc_matrix : np.ndarray, saved_dir_path : Path, dpi : int = 300, img_format : str = "png") -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(fc_matrix, cmap="RdBu_r", interpolation="nearest")  
    plt.colorbar()  
    plt.title("Heatmap")  
    plt.xlabel("X-axis")  
    plt.ylabel("Y-axis") 
    plt.savefig(saved_dir_path / "fc_matrix.png", dpi=dpi, format=img_format)
    plt.close()

def preprocess_fMRI_with_atlas(saved_dir_path : Path,  fMRI_path : Path, drop_frames : int, atlas_path : Path, lut_path   : Path = Brain_Atlas_Dir_Path.lut_txt_path) -> None:
    # Get regions names of atlas
    lut_dict = {}
    with lut_path.open("r") as f:
        lines = f.readlines()[1:] # the first row is "0 Unknown 25 5 25 0"
        for line in lines:
            line = line.strip().split()
            lut_dict[int(line[0])-1] = line[1] # start index from 0
    # Check the orientation, "LAS"
    assert get_orientation(path=fMRI_path) == get_orientation(path=atlas_path), f"The orientaion of {fMRI_path} is {get_orientation(path=fMRI_path)}, but the atlas is {get_orientation(path=atlas_path)}"

    # Step 1: Head Motion Correction
    corrected_func_path = saved_dir_path / "corrected_func.nii.gz"
    if not corrected_func_path.exists():
        corrected_func = head_motion_correction(func4d_path=fMRI_path)
        ants.image_write(image=corrected_func, filename=str(corrected_func_path))
        del corrected_func
    # change the dim[0] from 5 to 4
    adjust_dim_of_antsImage(corrected_func_path)
        
    # Step 2: Register to Atlas
    outprefix = "outprefix_of_registration_"
    aligned_func_path = saved_dir_path / "aligned_func.nii.gz"
    if not aligned_func_path.exists():
        moving_image = ants.image_read(filename=str(corrected_func_path))
        # Get the first valid frame (after skipping drop_frames frames)  
        reference_frame_idx = drop_frames  
        reference_frame = moving_image[:, :, :, reference_frame_idx:reference_frame_idx+1]  
        reference_temp_path = split_4dantsImageFrame_into_3d(image_4d=reference_frame) 
        # register only the reference frame to get the transformation  
        _, transforms = register(fixed_path=str(atlas_path), 
                                 moving_path=reference_temp_path,
                                 outprefix=outprefix)  

        # apply the same transformation to all frames (except the first drop_frames frames)  
        results_list = [] 
        for t in tqdm(range(moving_image.shape[-1]), desc=f"Registering {saved_dir_path.name}-func", leave=True):
            # delete the first drop_frames frames
            if t < reference_frame_idx:
                continue
            # t:t+1 cannot be t, and ":,:,:," cannot be "...,", otherwise the orientation will be wrong
            moving_image_t = moving_image[:, :, :, t:t+1]
            temp_path = split_4dantsImageFrame_into_3d(image_4d=moving_image_t)
            # register
            result = ants.apply_transforms(fixed=ants.image_read(filename=str(atlas_path)), 
                                            moving=ants.image_read(filename=temp_path),
                                            transformlist=transforms)
            results_list.append(result)
            # delete temporary file and variables
            Path(temp_path).unlink()
            del result, moving_image_t, temp_path
        ants.image_write(image=ants.merge_channels(image_list=results_list), filename=str(aligned_func_path))
        del moving_image, results_list
        for file in Path(".").glob(f"{outprefix}*"):
            file.unlink()
        del transforms
    # change the dim[0] from 5 to 4
    adjust_dim_of_antsImage(image_5d_path=aligned_func_path)

    # Step 3: Denoise
    denoised_func_path = saved_dir_path / "denoised_func.nii.gz"
    if not denoised_func_path.exists():
        print(f"Denoising {denoised_func_path.parent.name}: func")
        start_time = time.time()
        fwhm = float(re.search(pattern=r"(\d+(?:\.\d+)?)mm", string=str(atlas_path.name)).group(1))
        smoothed_img = image.smooth_img(  
            imgs=str(aligned_func_path),   
            fwhm=fwhm  # adjust the filtering width, mm  
        )
        smoothed_img.to_filename(str(denoised_func_path))  
        end_time = time.time()
        print(f"It took {end_time - start_time:.2f} seconds to denoise the {denoised_func_path.parent.name}.")
        del smoothed_img

    # Step 4: Functional connectivity
    fc_matrix_path = saved_dir_path / "features.npz"
    if not fc_matrix_path.exists():
        start_time = time.time()
        atlas = nib.load(filename=atlas_path)
        denoised_func = nib.load(filename=denoised_func_path)
        masker = maskers.NiftiLabelsMasker(labels_img=atlas, standardize="zscore_sample")
        time_series = masker.fit_transform(denoised_func) # shape=[len of series, num of regions]
        fc_matrix = connectome.ConnectivityMeasure(kind="correlation", standardize="zscore_sample").fit_transform([time_series])[0]
        np.fill_diagonal(fc_matrix, 0) # set the diagonal to 0 (1 -> 0)
        npz_data = {Experiment_Config.TS : time_series, Experiment_Config.FC : fc_matrix}
        np.savez_compressed(file=fc_matrix_path, **npz_data)
        plot_time_series(time_series=time_series, saved_dir_path=saved_dir_path, region_names=lut_dict)
        plot_fc_matrix(fc_matrix=fc_matrix, saved_dir_path=saved_dir_path)
        end_time = time.time()
        print(f"It took {end_time - start_time:.2f} seconds to calculate and plot the functional connectivity matrix.")

def process_mild_depression() -> int:
    """
    ds002748 + ds003007
    return: number of DPs - HCs
    """
    num_dp, num_hc = 0, 0
    ## ds002748
    raw_data_root_dir_path = Raw_Data_Dir_Path.ds002748_dir_path
    assert raw_data_root_dir_path.exists(), f"{raw_data_root_dir_path} dose not exist"
    # information
    participants_tsv_path = raw_data_root_dir_path / "participants.tsv"
    assert participants_tsv_path.exists(), f"{participants_tsv_path} does not exist"
    participants_info = pd.read_csv(participants_tsv_path, sep="\t")
    participants_info = participants_info[["participant_id", "age", "gender", "group"]]
    participants_info = participants_info.set_index("participant_id")
    participants_info = participants_info.to_dict(orient="index")
    for sub_id, info_dict in participants_info.items():
        info_dict["gender"] = Gender.Female if info_dict["gender"] == "f" else Gender.Male if info_dict["gender"] == "m" else None
        info_dict["group"] = Group.DP if info_dict["group"] == "depr" else Group.HC if info_dict["group"] == "control" else None
        assert info_dict["gender"] is not None, f"{sub_id} gender is None"
        assert info_dict["group"] is not None, f"{sub_id} group is None"
        num_dp += 1 if info_dict["group"] == Group.DP else 0
        num_hc += 1 if info_dict["group"] == Group.HC else 0
        # naming standardisation
        info_dict[Experiment_Config.A] = info_dict.pop("age")
        info_dict[Experiment_Config.G] = info_dict.pop("gender")
        info_dict[Experiment_Config.H] = Hand.X
        info_dict[Experiment_Config.T] = info_dict.pop("group")
        participants_info[sub_id] = info_dict
    # fMRI
    for sub_dir_path in raw_data_root_dir_path.glob("sub-*"):
        if sub_dir_path.is_dir():
            # saved dir path
            saved_dir_path = Running_File_Dir_Path.root_dir / raw_data_root_dir_path.name / sub_dir_path.name
            saved_dir_path.mkdir(parents=True, exist_ok=True)
            # information
            assert saved_dir_path.name in participants_info, f"{saved_dir_path.name} does not exist in participants_info"
            write_json(json_path=saved_dir_path / "information.json", dict_data=participants_info[saved_dir_path.name])
            # fMRI -> time series + functional connectivity
            func_file_path = list((sub_dir_path / "func").iterdir())
            assert len(func_file_path) == 1, f"{sub_dir_path} does not have func file or more than one"
            func_file_path = func_file_path[0]
            preprocess_fMRI_with_atlas(saved_dir_path=saved_dir_path, fMRI_path=func_file_path, drop_frames=5, atlas_path=Brain_Atlas_Dir_Path.atlas_1_nii_path)
            
    ## ds003007
    raw_data_root_dir_path = Raw_Data_Dir_Path.ds003007_dir_path
    assert raw_data_root_dir_path.exists(), f"{raw_data_root_dir_path} dose not exist"
    # information
    participants_tsv_path = raw_data_root_dir_path / "participants.tsv"
    assert participants_tsv_path.exists(), f"{participants_tsv_path} does not exist"
    participants_info = pd.read_csv(participants_tsv_path, sep="\t")
    participants_info = participants_info[["participant_id", "age", "gender"]]
    participants_info = participants_info.set_index("participant_id")
    participants_info = participants_info.to_dict(orient="index")
    for sub_id, info_dict in participants_info.items():
        info_dict["gender"] = Gender.Female if info_dict["gender"] == "f" else Gender.Male if info_dict["gender"] == "m" else None
        info_dict["group"] = Group.DP # all of them are depression patients
        assert info_dict["gender"] is not None, f"{sub_id} gender is None"
        assert info_dict["group"] is not None, f"{sub_id} group is None"
        num_dp += 1
        # naming standardisation
        info_dict[Experiment_Config.A] = info_dict.pop("age")
        info_dict[Experiment_Config.G] = info_dict.pop("gender")
        info_dict[Experiment_Config.H] = Hand.X
        info_dict[Experiment_Config.T] = info_dict.pop("group")
        participants_info[sub_id] = info_dict
    # fMRI
    for sub_dir_path in raw_data_root_dir_path.glob("sub-*"):
        if sub_dir_path.is_dir():
            sub_dir_path /= "ses-pre"
            # saved dir path
            saved_dir_path = Running_File_Dir_Path.root_dir / raw_data_root_dir_path.name / sub_dir_path.parent.name
            saved_dir_path.mkdir(parents=True, exist_ok=True)
            # information
            assert saved_dir_path.name in participants_info, f"{saved_dir_path.name} does not exist in participants_info"
            write_json(json_path=saved_dir_path / "information.json", dict_data=participants_info[saved_dir_path.name])
            # fMRI -> time series + functional connectivity
            func_file_path = list((sub_dir_path / "func").iterdir())
            assert len(func_file_path) == 1, f"{sub_dir_path} does not have func file or more than one"
            func_file_path = func_file_path[0]
            preprocess_fMRI_with_atlas(saved_dir_path=saved_dir_path, fMRI_path=func_file_path, drop_frames=5, atlas_path=Brain_Atlas_Dir_Path.atlas_1_nii_path)
    assert num_dp > num_hc, f"num_dp: {num_dp} <= num_hc: {num_hc}" # 80, 21
    return num_dp - num_hc

def process_heath_controls(selected_num : int) -> None:
    """
    Cambridge_Buckner
    """
    raw_data_root_dir_path = Raw_Data_Dir_Path.cambridge_dir_path
    assert raw_data_root_dir_path.exists(), f"{raw_data_root_dir_path} dose not exist"
    # information
    participants_info = {}
    with (raw_data_root_dir_path / "Cambridge_Buckner_demographics.txt").open("r") as f:
        for line in f:
            line = line.strip().split("\t")
            participants_info[line[0]] = {
                "group" : Group.HC,
                "age" : int(line[2]), 
                "gender" : Gender.Female if line[3] == "f" else Gender.Male if line[3] == "m" else None
            }
            assert participants_info[line[0]]["age"] is not None, f"{line[0]} age is None"
            assert participants_info[line[0]]["gender"] is not None, f"{line[0]} gender is None"
    with (raw_data_root_dir_path / "Cambridge_Buckner_handedness.txt").open("r") as f:
        for line in f:
            line = line.strip().split("\t")
            assert line[0] in participants_info, f"{line[0]} does not exist in participants_info"
            participants_info[line[0]]["hand"] = Hand.L if line[1] == "L" else Hand.R if line[1] == "R" else None
            assert participants_info[line[0]]["hand"] is not None, f"{line[0]} hand is None"
    # naming standardisation
    for sub_id, info_dict in participants_info.items():
        info_dict[Experiment_Config.A] = info_dict.pop("age")
        info_dict[Experiment_Config.G] = info_dict.pop("gender")
        info_dict[Experiment_Config.H] = info_dict.pop("hand")
        info_dict[Experiment_Config.T] = info_dict.pop("group")
        participants_info[sub_id] = info_dict
    # selected samples
    selected_sub_list = random.sample(list(participants_info.keys()), selected_num)
    # fMRI  
    for sub_dir_path in raw_data_root_dir_path.glob("sub*"):
        if sub_dir_path.is_dir() and sub_dir_path.name in selected_sub_list:
            # saved dir path
            saved_dir_path = Running_File_Dir_Path.root_dir / raw_data_root_dir_path.name / sub_dir_path.name
            saved_dir_path.mkdir(parents=True, exist_ok=True)
            # information
            assert saved_dir_path.name in participants_info, f"{saved_dir_path.name} does not exist in participants_info"
            write_json(json_path=saved_dir_path / "information.json", dict_data=participants_info[saved_dir_path.name])
            # fMRI -> time series + functional connectivity
            func_file_path = list((sub_dir_path / "func").iterdir())
            assert len(func_file_path) == 1, f"{sub_dir_path} does not have func file or more than one"
            func_file_path = func_file_path[0]
            preprocess_fMRI_with_atlas(saved_dir_path=saved_dir_path, fMRI_path=func_file_path, drop_frames=5, atlas_path=Brain_Atlas_Dir_Path.atlas_1_nii_path)

def split_folds() -> None:
    """
    Mild: ds002748, ds003007, Cambridge_Buckner
    Writed Json: {fold : {group : {task : list[str of path]}}}
    """
    def __split_list__(item_list : list[Path], n_splits : range = n_splits) -> dict[int, dict[str, list[str]]]:
        kf = KFold(n_splits=n_splits.stop-1, shuffle=False)
        random.shuffle(item_list)
        split_dict = {}
        fold = n_splits.start
        for train_index, test_index in kf.split(item_list):
            split_dict[fold] = {
                Experiment_Config.TRAIN : [item_list[i].as_posix() if isinstance(item_list[i], Path) else item_list[i] for i in train_index],
                Experiment_Config.TEST  : [item_list[i].as_posix() if isinstance(item_list[i], Path) else item_list[i] for i in test_index ]
            }
            fold += 1
        return split_dict
    
    def __group_split__(dataset_name_tuple : tuple[str]) -> dict[int, dict[str, list[str]]]:
        positive_samples, negative_samples = [], [] # positive_samples is depressive patients, negative_samples is health controls
        for dataset_name in dataset_name_tuple:
            processed_dir_path = Running_File_Dir_Path.root_dir / dataset_name
            assert processed_dir_path.exists(), f"{processed_dir_path} does not exist"
            for sub_dir_path in processed_dir_path.iterdir():
                information_path = sub_dir_path / "information.json"
                assert information_path.exists(), f"{information_path} does not exist"
                group = read_json(json_path=information_path)[Experiment_Config.T]
                if group == Group.DP:
                    positive_samples.append(sub_dir_path)
                elif group == Group.HC:
                    negative_samples.append(sub_dir_path)
                else:
                    raise ValueError(f"Unknown group {group} in {information_path}")
        positive_dict = __split_list__(item_list=positive_samples)
        negative_dict = __split_list__(item_list=negative_samples)
        return {k : {Group.DP : positive_dict[k], Group.HC : negative_dict[k]} for k in positive_dict}

    for (depr_type, name_tuple) in [
        (Experiment_Config.MILD , Mild_Config().dataset_name) , # mild
    ]:
        saved_json_path = Running_File_Dir_Path.split_dir_path / "".join([depr_type, ".json"])
        if not saved_json_path.exists():
            dict_data = __group_split__(dataset_name_tuple=name_tuple)
            write_json(json_path=saved_json_path, dict_data=dict_data)
        del saved_json_path

def main():
    # Yeo network of Brainnetome atlas
    get_yeo_network_of_brainnetome()
    # Mild depression
    selected_num = process_mild_depression()
    # Health Controls
    process_heath_controls(selected_num=selected_num)
    # Split folds
    split_folds()

if __name__ == "__main__":
    main()