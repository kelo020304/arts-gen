import json
import ipdb
import numpy as np
import os

def convert_to_floatvoxel(position,grid):
    indices=((np.array(position) + 0.5) * grid)

    indices=np.round(indices, 2)
    return indices.tolist()
def smart_round(arr, tol=1e-3):

    result = []
    for x in arr:
        nearest_int = round(x)
        diff = abs(x - nearest_int)

        if diff < tol:
            result.append(int(nearest_int))

        elif abs(diff - 0.1) < tol:
            result.append(round(x, 2))
        else:
            result.append(round(x, 2))
    return result
def process_lists(list_a, list_b):
    sum_a = sum(list_a)
    
    if sum_a >= 0:

        result_b = [min(list_b), max(list_b)]
    else:

        list_a = [-x for x in list_a]
        list_b = [-x for x in list_b]
        result_b = [min(list_b), max(list_b)]
    
    return list_a, result_b

def json_to_txt(json_file, txt_file):

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []

    lines.append(f"Name: {data['object_name']}")
    lines.append(f"Category: {data['category']}")
    lines.append(f"Dimension: {data['dimension']}")


    lines.append("Parts: ")
    for part in data["parts"]:
        label = part["label"]
        name = part["name"]
        desc = part["Basic_description"]
        material = part["material"]
        density = part["density"]
        affordance = max(min(part["priority_rank"],10),1)
        young = part["Young's Modulus (GPa)"]
        poisson = part["Poisson's Ratio"]
        lines.append(f"l_{label}: {name}, {affordance}, {material}, {density}, {young}, {poisson}, {desc}")


    lines.append("Group_info: ")
    group_info = data["group_info"]
    grid=32
    

    for k, v in group_info.items():
        if k == "0":
            parts = [f"l_{i}" for i in v]
            lines.append(f"group_{k}: {parts}; Type: E; Param: N/A")
        else:
            parts = [f"l_{i}" for i in v[0]]
            relative = f"group_{v[1]}"
            if v[3] == "A":
                joint_type = "Type: A relative to"
                extra = "Param: N/A"
            elif v[3] == "B":
                joint_type = "Type: B relative to"

                v[2][:3],v[2][6:]=process_lists(v[2][:3],v[2][6:])

                axis = np.array(v[2][:3], dtype=float)
                norm = np.linalg.norm(axis)
                if norm != 0:
                    axis_normalized = axis / norm
                else:
                    axis_normalized = axis  
                axis = smart_round(axis_normalized.tolist())
                
                movement = np.round(np.array(v[2][6:])*grid, 2).tolist()
                
                extra = f"Param: direction: {axis}, slide range (in voxel grid): {movement}"
            elif v[3] == "C":
                joint_type = "Type: C relative to"
                v[2][:3],v[2][6:]=process_lists(v[2][:3],v[2][6:])
         


                axis = np.array(v[2][:3], dtype=float)
                norm = np.linalg.norm(axis)
                if norm != 0:
                    axis_normalized = axis / norm
                else:
                    axis_normalized = axis  
                axis = smart_round(axis_normalized.tolist())

                
                position = convert_to_floatvoxel(v[2][3:6],grid)

                if len(np.where(np.array(axis)==1)[0])==1:
                    position[np.where(np.array(axis)==1)[0].item()]=0
                 
                movement = np.asarray(np.array(v[2][6:])*180, dtype=np.int64).tolist()

                
                
                
                extra = f"Param: direction: {axis}, axis position (in voxel grid): {position}, revolute range (degree): {movement}"
            elif v[3] == "D":
                joint_type = "Type: D relative to"
                position = convert_to_floatvoxel(v[2][3:6],grid)

                extra = f"Param: hinge position (in voxel grid): {position}"
            elif v[3] == "CB":
                joint_type = "Type: CB relative to"

 

                rotateaxis,rotaterange=process_lists(v[2][:3],v[2][6:8])
                axis = np.array(rotateaxis, dtype=float)
                norm = np.linalg.norm(axis)
                if norm != 0:
                    axis_normalized = axis / norm
                else:
                    axis_normalized = axis  
                axis = smart_round(axis_normalized.tolist())
                position = convert_to_floatvoxel(v[2][3:6],grid)

                if len(np.where(np.array(axis)==1)[0])==1:
                    position[np.where(np.array(axis)==1)[0].item()]=0
                
                movement=np.asarray(np.array(rotaterange)*180, dtype=np.int64).tolist()

                
                slideaxis,sliderange=process_lists(v[2][8:11],v[2][14:])
                axis2 = np.array(slideaxis, dtype=float)
                norm = np.linalg.norm(axis2)
                if norm != 0:
                    axis_normalized2 = axis2 / norm
                else:
                    axis_normalized2 = axis2  
                axis2 = smart_round(axis_normalized2.tolist())
                
                movement2 = np.round(np.array(sliderange)*grid, 2).tolist()

                extra = f"Param: axis direction: {axis}, axis position (in voxel grid): {position}, revolute range (degree): {movement}, slide direction: {axis2}, slide range (in voxel grid): {movement2}"

            lines.append(f"group_{k}: {parts}; {joint_type} {relative}; {extra}")

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    path='./tmp/finaljson'
    savepath='txt_rep_32_finetune'
    os.makedirs(savepath, exist_ok=True)
    namelist=os.listdir(path)
    for name in namelist:
        name=name[:-5]
        json_to_txt(os.path.join(path,name+'.json'), os.path.join(savepath,name+'.txt'))
        print(name)
