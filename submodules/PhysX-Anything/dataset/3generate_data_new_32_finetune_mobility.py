import os
import json
import numpy as np
from pathlib import Path
import ipdb
import argparse
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

import logging
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
def gen_conver(oriconv,question,answer):
    conversation1={}
    conversation1['from']="human"
    conversation1['value']=question
    conversation2={}
    conversation2['from']="gpt"
    conversation2['value']=answer
    oriconv.append(conversation1)
    oriconv.append(conversation2)
    return oriconv



def voxel_encode(voxels: np.ndarray, size: int = 32) -> np.ndarray:

    voxels = np.asarray(voxels, dtype=np.int64)
    assert voxels.ndim == 2 and voxels.shape[1] == 3, "voxels shape must be (N,3)"
    assert size == 32, "size=32（2^5）。"
    if (voxels < 0).any() or (voxels >= size).any():
        raise ValueError("xyz should in [0, 32).")

    x, y, z = voxels[:, 0], voxels[:, 1], voxels[:, 2]
    return (x << 10) | (y << 5) | z





def ints_to_space_separated_str(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.int64).ravel()
    return " ".join(map(str, arr))



def merge_adjacent_to_dash(s: str) -> str:

    if not s.strip():
        return ""

    nums = list(map(int, s.split()))

    nums = sorted(set(nums))

    result = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            result.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = n
    result.append(f"{start}-{prev}" if start != prev else f"{start}")
    return " ".join(result)





parser = argparse.ArgumentParser()
parser.add_argument("--ind", type=int, default=0)
parser.add_argument("--range", type=int, default=3000)
args = parser.parse_args()


    
alldata=[]

basepath='./tmp_mobility'
voxel_path=os.path.join(basepath,'partseg')
json_path=os.path.join(basepath,'finaljson')
namelist=os.listdir('./txt_rep_32_finetune_mobility_all')
namelist=namelist[args.ind*args.range:(args.ind+1)*args.range]
alllength=[]
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
logger = get_logger('info'+str(args.ind)+'.log',verbosity=1)
logger.info('start')

for name in namelist:
    name=name[:-4]
    with open(os.path.join(json_path,name+'.json'),'r') as f:
        jsondata=json.load(f)
    with open(os.path.join('txt_rep_32_finetune_mobility_all',name+'.txt'), "r", encoding="utf-8") as f:
        content = f.read()
    with open(os.path.join('overall_prompt.txt'), "r", encoding="utf-8") as f:
        basicqu = f.read()

    for part in range(len(jsondata['parts'])):

        for ind in range(0,25):
            dataseq={}
            dataseq['id']=name+'_'+str(ind).zfill(3)
            dataseq['image']=os.path.join(name,str(ind).zfill(3)+'.png')
            dataseq['conversations']=[]
            dataseq['data_source']='physx'

            
            


            gen_conver(dataseq['conversations'],\
                    "<image>\n"+basicqu,\
                    content
            )

            

            anspart1='Part name: '+jsondata['parts'][part]['name']+'. '+'Material: '+jsondata['parts'][part]['material']+'. '+'density: '+jsondata['parts'][part]['density']+'. '+'Affordance: '+str(jsondata['parts'][part]['priority_rank'])+'. '+'Young: '+str(jsondata['parts'][part]["Young's Modulus (GPa)"])+'. '+'Poisson: '+str(jsondata['parts'][part]["Poisson's Ratio"])+'.\n'
            anspart2='Basic_description: '+jsondata['parts'][part]['Basic_description']+' '+'Functional_description: '+jsondata['parts'][part]['Functional_description']+' '+'Movement_description: '+jsondata['parts'][part]['Movement_description']+''

            voxeldata=np.load(os.path.join(voxel_path,name,'32','ind_'+str(part)+'.npy'))
            
            voxeldata=voxeldata[np.lexsort((voxeldata[:,2], voxeldata[:,1], voxeldata[:,0]))]
            dataseq['meshlength']=len(voxeldata)

            

            



            idx = voxel_encode(voxeldata)  
            


            s = ints_to_space_separated_str(idx)


            s_dash = merge_adjacent_to_dash(s)

            
        
            
            gen_conver(dataseq['conversations'],\
                    "Based on the structured description of l_"+str(part)+", generate its 3D voxel grid in the following format (voxel grid=32, use numbers from 0 to 32767, merge maximal consecutive runs: 199...216 -> 199-216): 184 198 199-216 230-237...",\
                    s_dash
            )
            alllength.append(len(tokenizer(s_dash)["input_ids"]))

            

            


            alldata.append(dataseq)
            
    logger.info(name)

with open(os.path.join('im_data_obj_sort_new_32_finetune_final','training_set_'+str(args.ind)+'_randompart_mobility_all.json'), 'w') as json_file:
    json.dump(alldata, json_file, indent=4)   

plt.hist(np.array(alllength), bins=100, color='skyblue')
plt.savefig(str(args.ind)+'.png')

