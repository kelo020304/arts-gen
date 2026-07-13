import open3d as o3d
import numpy as np
import ipdb
import trimesh
import os
import json
import pickle
import logging
import argparse

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

def movtran(orimesh,offset,scale):
    orimesh.apply_transform(trimesh.transformations.scale_matrix(scale))
    orimesh.apply_translation([offset[0],offset[1],offset[2]])
    rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    orimesh.apply_transform(rotation_matrix)
    return orimesh
def transfer(point):
    ph = np.append(point, 1.0)
    p_new = (M @ ph)[:3]
    return p_new
def generate_voxel(mesh,voxel_define=64):


    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices) #

    point_cloud = mesh.sample_points_poisson_disk(number_of_points=81920,init_factor=3,pcl=None)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(point_cloud, voxel_size=1/voxel_define, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < voxel_define), "Some vertices are out of bounds"

    vertices = (vertices + 0.5) / voxel_define - 0.5

    indices = ((vertices + 0.5) * voxel_define)
    indices = np.asarray(indices, dtype=np.int64)
    return indices,vertices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ind", type=int, default=0)
    parser.add_argument("--range", type=int, default=100)
    args = parser.parse_args()

    basepath='./PhysXNet/'      # dataset path
    os.makedirs('tmp/', exist_ok=True)
    indexlist=os.listdir(os.path.join(basepath,'partseg'))
    logger = get_logger('./tmp/exp_1render'+str(args.ind)+'.log',verbosity=1)
    logger.info('start')

    indexlist=indexlist[args.ind*args.range:(args.ind+1)*args.range]


    for index in indexlist:
        objpath=os.path.join(basepath,'partseg',index,'objs')
        tmpdir=os.path.join('tmp/','partseg',index)
        os.makedirs(tmpdir, exist_ok=True)
        os.makedirs(os.path.join('tmp/','finaljson'), exist_ok=True)
        jsonpath=os.path.join(basepath,'finaljson',index+'.json')
        with open(jsonpath,'r') as f:
            jsondata=json.load(f)

        len(jsondata['parts'])*2+2
        if os.path.exists(os.path.join(tmpdir,'64')):
            if len(os.listdir(os.path.join(tmpdir,'64')))==len(jsondata['parts'])*2+2:
                logger.info('skip: '+index)

                continue
        needrefineb={}
        needrefinec={}
        needrefined={}
        needrefinecb={}
        newrefineb={}
        newrefinec={}
        newrefined={}
        newrefinecb={}
        if len(jsondata['group_info'])>1:
            for groupind in range(1,len(jsondata['group_info'])):
                if jsondata['group_info'][str(groupind)][-1]=='B':
                    needrefineb[str(groupind)]=jsondata['group_info'][str(groupind)][-2][:3]

                if jsondata['group_info'][str(groupind)][-1]=='C':
                    needrefinec[str(groupind)]=jsondata['group_info'][str(groupind)][-2][:6]

                if jsondata['group_info'][str(groupind)][-1]=='D':
                    needrefined[str(groupind)]=jsondata['group_info'][str(groupind)][-2][3:6]

                if jsondata['group_info'][str(groupind)][-1]=='CB':

                    needrefinecb[str(groupind)]=jsondata['group_info'][str(groupind)][-2][:6]+jsondata['group_info'][str(groupind)][-2][8:11]




        namelist=os.listdir(objpath)
        namelist = sorted(namelist, key=lambda x: int(x.split('.')[0]))

        orimesh=trimesh.Trimesh([])
        for name in namelist:
            oripart=trimesh.load(os.path.join(objpath,name))
            orimesh = trimesh.util.concatenate([orimesh,oripart])
        orimesh.merge_vertices()





        bbox_max=np.array(orimesh.vertices).max(0)
        bbox_min=np.array(orimesh.vertices).min(0)
        scale = 1 / max(bbox_max - bbox_min)
        offset = -(bbox_min + bbox_max) / 2
        movtran(orimesh,offset,scale)
        M_scale = trimesh.transformations.scale_matrix(scale)
        M_trans = trimesh.transformations.translation_matrix(offset)
        M_rot   = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        M = M_rot @ M_trans @ M_scale

        if len(needrefineb)>0:
            for groupind in list(needrefineb.keys()):
                point1=[0,0,0]
                point2=needrefineb[groupind][:3]

                jsondata['group_info'][groupind][-2][:3]=transfer(point2)-transfer(point1)
                
                
        if len(needrefinec)>0:
            for groupind in list(needrefinec.keys()):
                point1=[0,0,0]
                point2=needrefinec[groupind][:3]
                point3=needrefinec[groupind][3:6]
                
                jsondata['group_info'][groupind][-2][:3]=transfer(point2)-transfer(point1)
                jsondata['group_info'][groupind][-2][3:6]=transfer(point3)



        if len(needrefined)>0:
            for groupind in list(needrefined.keys()):
                point1=needrefined[groupind][:3]

                jsondata['group_info'][groupind][-2][3:6]=transfer(point1)
                

        if len(needrefinecb)>0:
            for groupind in list(needrefinecb.keys()):
                point1=[0,0,0]
                point2=needrefinecb[groupind][:3]
                point3=needrefinecb[groupind][3:6]
                point4=needrefinecb[groupind][6:]

                jsondata['group_info'][groupind][-2][:3]=transfer(point2)-transfer(point1)
                jsondata['group_info'][groupind][-2][3:6]=transfer(point3)
                jsondata['group_info'][groupind][-2][8:11]=transfer(point4)-transfer(point1)

        
        with open(os.path.join('tmp/','finaljson',index+'.json'),'w') as file:
            json.dump(jsondata, file, indent=4, ensure_ascii=False)



        resolist=[16,32,64]

        for res in resolist:
            savepath=os.path.join(tmpdir,str(res))
            os.makedirs(savepath, exist_ok=True)

            alldict_ind={}
            alldict_vert={}

            allind = np.empty((0, 3))  
            for name in namelist:
                oripart1=trimesh.load(os.path.join(objpath,name)) 
                movtran(oripart1,offset,scale)
                oripart1.export(os.path.join(savepath,'mesh_new_'+name.split('.')[0]+'.ply'))
                part=o3d.io.read_triangle_mesh(os.path.join(savepath,'mesh_new_'+name.split('.')[0]+'.ply'))
                indices1,vertices1=generate_voxel(part,res)
                alldict_ind[name.split('.')[0]]=indices1
                alldict_vert[name.split('.')[0]]=vertices1
                #ipdb.set_trace()
                np.save(os.path.join(savepath,'ind_'+name.split('.')[0]+'.npy'),indices1)
                allind=np.concatenate([allind,indices1],0)
                logger.info(index+'_'+name+": "+str(indices1.shape))


            with open(os.path.join(savepath,'alldict_vert.pkl'), "wb") as f:
                pickle.dump(alldict_vert, f)

            np.save(os.path.join(savepath,'allind.npy'),allind)
            logger.info(index+": "+str(allind.shape))

