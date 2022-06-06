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

# PostProcessingStack

import torch

# from ast import increment_lineno
import numpy as np
import cv2
# import matplotlib.pyplot as plt

from wand.image import Image
from wand.display import display

# from filmnoise.utilities import load_image, resize_image
from filmnoise.noise_image import NoiseImage

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import PIL
from text_utils import ImageText

import os
from os import path

from zipfile import ZipFile

import imageio as iio
from pathlib import Path

root_path = os.getcwd()

# os.system('pip install pillow_lut')
# os.system('pip install pyglet')

from pillow_lut import load_hald_image, load_cube_file, rgb_color_enhance
import pyglet

# class NoneClass:
#     pass

def post_process(source_path, post_path, post_caption, colormap, grain_size = 5):
    img = Image.open(source_path)
    postdir = os.path.dirname(post_path)
    filename = os.path.basename(post_path)
    
    # Iterate over images in a folder TODO WIP make it do something useful
    def postprocess_files_in_dir(path_to_image_folder):
        images = list()
        for file in Path(f"{path_to_image_folder}").iterdir():
            im = iio.imread(file)
            images.append(im)

    # Iterate over frames in a movie TODO WIP make it do something useful
    def postprocess_movie(path_to_movie):
        reader = iio.get_reader(path_to_movie)
        fps = reader.get_meta_data()['fps']
        writer = iio.get_writer(f'{path_to_movie}', fps=fps)
        for i, im in enumerate(reader):
            print('Mean of frame %i is %1.1f' % (i, im.mean()))
        
    # Iterate over images in a ZIP archive. Assuming there is only image files in the ZIP archive, you can iterate over them with a simple script like the one below. TODO make it do something useful
    def postprocess_zip(path_to_zip):
        images = list()
        with ZipFile(path_to_zip) as zf:
            for name in zf.namelist():
                im = iio.imread(name)
                images.append(im)

    def apply_LUT(img, post_path, colormap, brightness=0, exposure=0.2, contrast=0.1, warmth=0, saturation=0, vibrance=0.5, hue=0, gamma=1.3,linear=False, cls=ImageFilter.Color3DLUT):
        # map = load_hald_image(colormap)
        map = load_cube_file(colormap)
        lut = rgb_color_enhance(map, brightness, exposure, contrast, warmth, saturation, vibrance, hue, gamma, linear, cls)
        # img = Image.open(source_path)
        img.filter(lut).save(post_path)

    ## TODO Convert to cv2?
    def add_caption(source_path, post_path, post_caption, anchor_x=0, anchor_y=0, text_color=(0,0,0), stroke_size=2, stroke_fill_color=(255, 255, 255), background_color=(255,255,255)):
        # post_img = img
        # width_og, height_og = im_og.size
        # img = Image.open(f'{source_path}')
        d1 = ImageDraw.Draw(img)
        # captionFont = ImageFont.load_default()
        captionFont = ImageFont.truetype(f'{root_path}/PostProcessingStack/fonts/Bebas-Regular.ttf', 28)
        w, h = captionFont.getsize(post_caption)
        d1.rectangle((anchor_x, anchor_y, anchor_x + w, anchor_y + h), fill=background_color)
        d1.text((0, 0), f"{post_caption}", font=captionFont, fill =text_color, stroke_width=stroke_size, stroke_fill=stroke_fill_color)
        img.save(post_path)
        img.show()
    
    def add_film_grain(img, post_path, mix=0.5, grain_size = 5):
        
        postdir = os.path.dirname(post_path)
        filename = os.path.basename(post_path)

        im_path = os.path.abspath(img)
        im = np.array(Image.open(im_path)).astype('float')
        im = im.mean(axis=2)  # convert to bw
        n = NoiseImage(im, grain_size)
        n.process()
        
        color_im = np.array(Image.open(img)).astype('float')/255.0
        x= color_im*(0.5+0.5*np.tile(n.resized_out[:,:,None],(1,1,1)))
        x.shape
        
        formatted = (x * 255 / np.max(x)).astype('uint8')
        img = Image.fromarray(formatted)
        # plt.imshow(img)
        img.save(f'{postdir}/grain-{filename}')
        
   
    # def add_ffmpeg_film_grain(movie):
    #     ffmpeg -i "HD Splice 1080p No Grain.mkv" -i "HD Splice 1080p No Grain.mkv" -filter_complex " color=black:d=3006.57:s=3840x2160:r=24000/1001, geq=lum_expr=random(1)*256:cb=128:cr=128, deflate=threshold0=15, dilation=threshold0=10, eq=contrast=3, cale=1920x1080 [n]; [0] eq=saturation=0,geq=lum='0.15*(182-abs(75-lum(X,Y)))':cb=128:cr=128 [o]; [n][o] blend=c0_mode=multiply,negate [a]; color=c=black:d=3006.57:s=1920x1080:r=24000/1001 [b]; [1][a] alphamerge [c]; b][c] overlay,ass=Subs.ass" -c:a copy -c:v libx264 -tune grain -preset veryslow -crf 12 -y Output-1080p-Grain.mkv
    
    def get_fps_from_movie(movie):
        
        reader = iio.get_reader(f'imageio:{movie}')
        fps = reader.get_meta_data()['fps']
    
    def create_depthmap(img, post_path):
        postdir = os.path.dirname(post_path)
        filename = os.path.basename(post_path)
        # title = "MiDaS"
        # description = "Gradio demo for MiDaS v2.1 which takes in a single image for computing relative depth. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
        # article = "<p style='text-align: center'><a href='https://arxiv.org/abs/1907.01341v3'>Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer</a> | <a href='https://github.com/intel-isl/MiDaS'>Github Repo</a></p>"
	
        # torch.hub.download_url_to_file('https://images.unsplash.com/photo-1437622368342-7a3d73a34c8f', 'turtle.jpg')
        # torch.hub.download_url_to_file('https://images.unsplash.com/photo-1519066629447-267fffa62d4b', 'lions.jpg')

        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")

        use_large_model = True

        if use_large_model:
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        else:
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            
        device = "cpu"
        midas.to(device)

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if use_large_model:
            transform = midas_transforms.default_transform
        else:
            transform = midas_transforms.small_transform
        
        
        cv_image = np.array(img) 
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        output = prediction.cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype('uint8')
        img = Image.fromarray(formatted)
        # plt.imshow(img)
        img.save(f'{postdir}/depth/{filename}-depthmap.png')
        # img.show()
        

    # Apply Post Processing effects
    
    create_depthmap(img, post_path)
    apply_LUT(img, post_path, colormap)
    add_film_grain(f'{postdir}/{filename}', post_path, 0.5, grain_size = 10)
    # add_caption(img, post_path, post_caption)
