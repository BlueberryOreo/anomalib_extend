"""Custom Folder Dataset.

This script creates a custom dataset from a folder.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional, Tuple, Union

import albumentations as A
from pandas import DataFrame
from torchvision.datasets.folder import IMG_EXTENSIONS

from anomalib_extend.data.base import AnomalibDataModule, AnomalibDataset
from anomalib_extend.data.task_type import TaskType
from anomalib_extend.data.utils import (
    InputNormalizationMethod,
    Split,
    TestSplitMode,
    ValSplitMode,
    get_transforms,
)


def _check_and_convert_path(path: Union[str, Path]) -> Path:
    """Check an input path, and convert to Pathlib object.

    Args:
        path (Union[str, Path]): Input path.

    Returns:
        Path: Output path converted to pathlib object.
    """
    if not isinstance(path, Path):
        path = Path(path)
    return path


def _prepare_files_labels(
    path: Union[str, Path], path_type: str, extensions: Optional[Tuple[str, ...]] = None
) -> Tuple[list, list]:
    """Return a list of filenames and list corresponding labels.

    Args:
        path (Union[str, Path]): Path to the directory containing images.
        path_type (str): Type of images in the provided path ("normal", "abnormal", "normal_test")
        extensions (Optional[Tuple[str, ...]], optional): Type of the image extensions to read from the
            directory.

    Returns:
        List, List: Filenames of the images provided in the paths, labels of the images provided in the paths
    """
    path = _check_and_convert_path(path)
    if extensions is None:
        extensions = IMG_EXTENSIONS

    if isinstance(extensions, str):
        extensions = (extensions,)

    filenames = [f for f in path.glob(r"**/*") if f.suffix in extensions and not f.is_dir()]
    if len(filenames) == 0:
        raise RuntimeError(f"Found 0 {path_type} images in {path}")

    labels = [path_type] * len(filenames)

    return filenames, labels


def _resolve_path(folder: Union[Path, str], root: Optional[Union[Path, str]] = None) -> Path:
    """Combines root and folder and returns the absolute path.

    This allows users to pass either a root directory and relative paths, or absolute paths to each of the
    image sources. This function makes sure that the samples dataframe always contains absolute paths.

    Args:
        folder (Optional[Union[Path, str]]): Folder location containing image or mask data.
        root (Optional[Union[Path, str]]): Root directory for the dataset.
    """
    folder = Path(folder)
    if folder.is_absolute():
        # path is absolute; return unmodified
        path = folder
    # path is relative.
    elif root is None:
        # no root provided; return absolute path
        path = folder.resolve()
    else:
        # root provided; prepend root and return absolute path
        path = (Path(root) / folder).resolve()
    return path


def make_folder_dataset(
    normal_dir: Union[str, Path],
    root: Optional[Union[str, Path]] = None,
    abnormal_dir: Optional[Union[str, Path]] = None,
    normal_test_dir: Optional[Union[str, Path]] = None,
    mask_dir: Optional[Union[str, Path]] = None,
    split: Optional[Union[Split, str]] = None,
    extensions: Optional[Tuple[str, ...]] = None,
):
    """Make Folder Dataset.

    Args:
        normal_dir (Union[str, Path]): Path to the directory containing normal images.
        root (Optional[Union[str, Path]]): Path to the root directory of the dataset.
        abnormal_dir (Optional[Union[str, Path]], optional): Path to the directory containing abnormal images.
        normal_test_dir (Optional[Union[str, Path]], optional): Path to the directory containing
            normal images for the test dataset. Normal test images will be a split of `normal_dir`
            if `None`. Defaults to None.
        mask_dir (Optional[Union[str, Path]], optional): Path to the directory containing
            the mask annotations. Defaults to None.
        split (Optional[Union[Split, str]], optional): Dataset split (ie., Split.FULL, Split.TRAIN or Split.TEST).
            Defaults to None.
        extensions (Optional[Tuple[str, ...]], optional): Type of the image extensions to read from the
            directory.

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """
    normal_dir = _resolve_path(normal_dir, root)
    abnormal_dir = _resolve_path(abnormal_dir, root) if abnormal_dir is not None else None
    normal_test_dir = _resolve_path(normal_test_dir, root) if normal_test_dir is not None else None
    mask_dir = _resolve_path(mask_dir, root) if mask_dir is not None else None
    assert normal_dir.is_dir(), "A folder location must be provided in normal_dir."

    filenames = []
    labels = []
    dirs = {"normal": normal_dir}

    if abnormal_dir:
        dirs = {**dirs, **{"abnormal": abnormal_dir}}

    if normal_test_dir:
        dirs = {**dirs, **{"normal_test": normal_test_dir}}

    for dir_type, path in dirs.items():
        filename, label = _prepare_files_labels(path, dir_type, extensions)
        filenames += filename
        labels += label

    samples = DataFrame({"image_path": filenames, "label": labels, "mask_path": ""})

    # Create label index for normal (0) and abnormal (1) images.
    samples.loc[(samples.label == "normal") | (samples.label == "normal_test"), "label_index"] = 0
    samples.loc[(samples.label == "abnormal"), "label_index"] = 1
    samples.label_index = samples.label_index.astype(int)

    # If a path to mask is provided, add it to the sample dataframe.
    if mask_dir is not None:
        mask_dir = _check_and_convert_path(mask_dir)
        for index, row in samples.iterrows():
            if row.label_index == 1:
                rel_image_path = row.image_path.relative_to(abnormal_dir)
                samples.loc[index, "mask_path"] = str(mask_dir / rel_image_path)

        # make sure all the files exist
        # samples.image_path does NOT need to be checked because we build the df based on that
        assert samples.mask_path.apply(
            lambda x: Path(x).exists() if x != "" else True
        ).all(), f"missing mask files, mask_dir={mask_dir}"

    # Ensure the pathlib objects are converted to str.
    # This is because torch dataloader doesn't like pathlib.
    samples = samples.astype({"image_path": "str"})

    # Create train/test split.
    # By default, all the normal samples are assigned as train.
    #   and all the abnormal samples are test.
    samples.loc[(samples.label == "normal"), "split"] = "train"
    samples.loc[(samples.label == "abnormal") | (samples.label == "normal_test"), "split"] = "test"

    # Get the data frame for the split.
    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class FolderDataset(AnomalibDataset):
    """Folder dataset.

    Args:
        task (TaskType): Task type. (``classification``, ``detection`` or ``segmentation``).
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        split (Optional[Union[Split, str]]): Fixed subset split that follows from folder structure on file system.
            Choose from [Split.FULL, Split.TRAIN, Split.TEST]
        normal_dir (Union[str, Path]): Path to the directory containing normal images.
        root (Optional[Union[str, Path]]): Root folder of the dataset.
        abnormal_dir (Optional[Union[str, Path]], optional): Path to the directory containing abnormal images.
        normal_test_dir (Optional[Union[str, Path]], optional): Path to the directory containing
            normal images for the test dataset. Defaults to None.
        mask_dir (Optional[Union[str, Path]], optional): Path to the directory containing
            the mask annotations. Defaults to None.

        extensions (Optional[Tuple[str, ...]], optional): Type of the image extensions to read from the
            directory.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.

    Raises:
        ValueError: When task is set to classification and `mask_dir` is provided. When `mask_dir` is
            provided, `task` should be set to `segmentation`.
    """

    def __init__(
        self,
        task: TaskType,
        transform: A.Compose,
        normal_dir: Union[str, Path],
        root: Optional[Union[str, Path]] = None,
        abnormal_dir: Optional[Union[str, Path]] = None,
        normal_test_dir: Optional[Union[str, Path]] = None,
        mask_dir: Optional[Union[str, Path]] = None,
        split: Optional[Union[Split, str]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__(task, transform)

        self.split = split
        self.root = root
        self.normal_dir = normal_dir
        self.abnormal_dir = abnormal_dir
        self.normal_test_dir = normal_test_dir
        self.mask_dir = mask_dir
        self.extensions = extensions

    def _setup(self):
        """Assign samples."""
        self.samples = make_folder_dataset(
            root=self.root,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            split=self.split,
            extensions=self.extensions,
        )


class Folder(AnomalibDataModule):
    """Folder DataModule.

    Args:
        normal_dir (Union[str, Path]): Name of the directory containing normal images.
            Defaults to "normal".
        root (Optional[Union[str, Path]]): Path to the root folder containing normal and abnormal dirs.
        abnormal_dir (Optional[Union[str, Path]]): Name of the directory containing abnormal images.
            Defaults to "abnormal".
        normal_test_dir (Optional[Union[str, Path]], optional): Path to the directory containing
            normal images for the test dataset. Defaults to None.
        mask_dir (Optional[Union[str, Path]], optional): Path to the directory containing
            the mask annotations. Defaults to None.
        normal_split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.2.
        extensions (Optional[Tuple[str, ...]], optional): Type of the image extensions to read from the
            directory. Defaults to None.
        image_size (Optional[Union[int, Tuple[int, int]]], optional): Size of the input image.
            Defaults to None.
        center_crop (Optional[Union[int, Tuple[int, int]]], optional): When provided, the images will be center-cropped
            to the provided dimensions.
        normalize (bool): When True, the images will be normalized to the ImageNet statistics.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        test_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        task (TaskType, optional): Task type. Could be ``classification``, ``detection`` or ``segmentation``.
            Defaults to segmentation.
        transform_config_train (Optional[Union[str, A.Compose]], optional): Config for pre-processing
            during training.
            Defaults to None.
        transform_config_val (Optional[Union[str, A.Compose]], optional): Config for pre-processing
            during validation.
            Defaults to None.
        test_split_mode (TestSplitMode): Setting that determines how the testing subset is obtained.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
        seed (Optional[int], optional): Seed used during random subset splitting.
    """

    def __init__(
        self,
        normal_dir: Union[str, Path],
        root: Optional[Union[str, Path]] = None,
        abnormal_dir: Optional[Union[str, Path]] = None,
        normal_test_dir: Optional[Union[str, Path]] = None,
        mask_dir: Optional[Union[str, Path]] = None,
        normal_split_ratio: float = 0.2,
        extensions: Optional[Tuple[str]] = None,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        center_crop: Optional[Union[int, Tuple[int, int]]] = None,
        normalization: Union[InputNormalizationMethod, str] = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType = TaskType.SEGMENTATION,
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_eval: Optional[Union[str, A.Compose]] = None,
        test_split_mode: TestSplitMode = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.normal_split_ratio = normal_split_ratio

        transform_train = get_transforms(
            config=transform_config_train,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )
        transform_eval = get_transforms(
            config=transform_config_eval,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )

        self.train_data = FolderDataset(
            task=task,
            transform=transform_train,
            split=Split.TRAIN,
            root=root,
            normal_dir=normal_dir,
            abnormal_dir=abnormal_dir,
            normal_test_dir=normal_test_dir,
            mask_dir=mask_dir,
            extensions=extensions,
        )

        self.test_data = FolderDataset(
            task=task,
            transform=transform_eval,
            split=Split.TEST,
            root=root,
            normal_dir=normal_dir,
            abnormal_dir=abnormal_dir,
            normal_test_dir=normal_test_dir,
            mask_dir=mask_dir,
            extensions=extensions,
        )
