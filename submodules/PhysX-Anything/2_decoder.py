import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import argparse
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import ipdb
import numpy as np
import torch
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument("--image_name", type=str, default=None, help="单张图片文件名，如 0.png")
args = parser.parse_args()

basepath='./demo'
filepath='./test_demo'

if args.image_name:
    args.image_name = os.path.basename(args.image_name)
    image_path_check = os.path.join(basepath, args.image_name)
    if not os.path.exists(image_path_check):
        raise FileNotFoundError(f"Image not found: {image_path_check}")
    stem = os.path.splitext(os.path.basename(args.image_name))[0]
    qwenpath_check = os.path.join(filepath, stem, 'allind.npy')
    if not os.path.exists(qwenpath_check):
        raise FileNotFoundError(f"allind.npy not found: {qwenpath_check}, run 1_vlm_demo.py first")
    namelist = [args.image_name]
    print(f"Processing single image: {args.image_name}")
else:
    namelist = [f for f in os.listdir(basepath) if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg'}]

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("./pretrain/decoder")
pipeline.cuda()


for name in namelist:
    image = Image.open(os.path.join(basepath,name))
    stem = os.path.splitext(name)[0]
    qwenpath=os.path.join(filepath,stem)

    if os.path.exists(os.path.join(qwenpath,'allind.npy')):
        newcoords=np.load(os.path.join(qwenpath,'allind.npy'))
        
        size=32
        resolution=64

        newcoords=newcoords+32-(size)//2
        
        ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
        ss[:, newcoords[:, 0], newcoords[:, 1], newcoords[:, 2]] = 1
        ss=ss.cuda().float().unsqueeze(0)



        outputs = pipeline.run_control(ss,image,seed=1,formats=['mesh', 'gaussian'])

        # Free pipeline GPU memory before texture baking
        pipeline.cpu()
        torch.cuda.empty_cache()

        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.5,          # Ratio of triangles to remove in the simplification process
            texture_size=512,       # Size of the texture used for the GLB (reduced from 1024 to save VRAM)
        )


        

        
        glb.export(os.path.join(qwenpath,'sample.glb'))
    elif args.image_name:
        raise FileNotFoundError(f"allind.npy not found in {qwenpath}, run 1_vlm_demo.py first")
