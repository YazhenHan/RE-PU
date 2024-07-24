#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: visualization.py
@Time: 2020/1/2 10:26 AM
"""

import os
import numpy as np
import itertools
import argparse
import mitsuba as mi
from tqdm import tqdm


def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    return result

xml_head = \
"""
<scene version="0.5.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1600"/>
            <integer name="height" value="1200"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="0.015"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
            <scale value="0.7"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""

def colormap(x,y,z):
    vec = np.array([x,y,z])
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]

def mitsuba(pcl, path, clr=None):
    xml_segments = [xml_head]
    # pcl = standardize_bbox(pcl, 2048)
    # pcl = pcl - np.expand_dims(np.mean(pcl, axis=0), 0)  # center
    # dist = np.max(np.sqrt(np.sum(pcl ** 2, axis=1)), 0)
    # pcl = pcl / dist  # scale
    pcl = pcl[:,[2,0,1]]
    pcl[:,0] *= -1
    h = np.min(pcl[:,2])
    if clr == "plane":
        clrgrid = [[0, 1, 45], [1, 0, 45]]
        b = np.linspace(*clrgrid[0])
        c = np.linspace(*clrgrid[1])
        color_all = np.array(list(itertools.product(b, c)))
        color_all = np.concatenate((np.linspace(1, 0, 2025)[..., np.newaxis], color_all), axis=1)
    elif clr == "sphere":
        color_all = np.load("sphere.npy")
        color_all = (color_all + 0.3) / 0.6
    elif clr == "gaussian":
        color_all = np.load("gaussian.npy")
        color_all = (color_all + 0.3) / 0.6
    for i in range(pcl.shape[0]):
        if clr == None:
            color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5)
        elif clr in ["plane", "sphere", "gaussian"]:
            color = color_all[i]
        else:
            color = clr
        if h < -0.25:
            xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2]-h-0.6875, *color))
        else:
            xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
    xml_segments.append(xml_tail)
    xml_content = str.join('', xml_segments)
    with open(path, 'w') as f:
        f.write(xml_content)

def draw_image(xml_path):
    mi.set_variant('scalar_rgb')
    img = mi.render(mi.load_file(xml_path))
    mi.util.write_bitmap('.'.join(xml_path.split('.')[:-1]) + ".png", img)


if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='RE-PU Visualization')
    parser.add_argument('--exp_name', type=str, default=None, metavar='N',
                        help='Name of the experiment')
    args = parser.parse_args()

    print(str(args))
    snapshot_root = f'snapshot/{args.exp_name}/' 
    images_dir = snapshot_root + "images/"
    os.makedirs(images_dir, exist_ok=True)
    upsampling_dir = snapshot_root + "upsampling/"
    
    for file in tqdm(os.listdir(upsampling_dir)):
        save_path = os.path.join(images_dir, file[:-4] + ".xml")
        points = np.loadtxt(upsampling_dir + file).astype(np.float32)
        mitsuba(points, save_path)
        draw_image(save_path)