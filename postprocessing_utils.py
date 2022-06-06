#!/usr/bin/env python
# coding: utf-8

# @title Licensed under the MIT License
# Copyright (c) 2022 Clement-Brice Girault 

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# PostProcessingUtils

def get_image_names(folder: Path) -> List[Path]:
    """
    Find the names and paths of all of the images in a folder.

    Args:
        folder: Folder containing images (assume folder only contains images).

    Returns:
        A list of the names with paths of the images in a folder.
    """
    return sorted([
        Path(f.name) for f in folder.iterdir() if ((
            folder.joinpath(f.name).is_file()) and (".DS_Store" not in f.name))
    ],
                  key=str)


def get_image_paths(folder: Path) -> List[Path]:
    """
    Find the full paths of the images in a folder.

    Args:
        folder: Folder containing images (assume folder only contains images).

    Returns:
        A list of the full paths to the images in the folder.
    """
    return sorted([
        folder.joinpath(f.name) for f in folder.iterdir() if ((
            folder.joinpath(f.name).is_file()) and (".DS_Store" not in f.name))
    ],
                  key=str)


def get_subfolder_paths(folder: Path) -> List[Path]:
    """
    Find the paths of subfolders.

    Args:
        folder: Folder to look for subfolders in.

    Returns:
        A list containing the paths of the subfolders.
    """
    return sorted([
        folder.joinpath(f.name) for f in folder.iterdir()
        if ((folder.joinpath(f.name).is_dir()) and (".DS_Store" not in f.name))
    ],
                  key=str)


def get_all_image_paths(master_folder: Path) -> List[Path]:
    """
    Finds all image paths in subfolders.

    Args:
        master_folder: Root folder containing subfolders.

    Returns:
        A list of the paths to the images found in the folder.
    """
    all_paths = []
    subfolders = get_subfolder_paths(folder=master_folder)
    if len(subfolders) > 1:
        for subfolder in subfolders:
            all_paths += get_image_paths(folder=subfolder)
    else:
        all_paths = get_image_paths(folder=master_folder)
    return all_paths
