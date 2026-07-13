
import cv2  
import base64
import time
from openai import OpenAI
import os
import imageio

import json
def readvideo(path,filename):
    videoframe = []
    
    video = cv2.VideoCapture(os.path.join(path,filename+'.mp4'))

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_jump = int(fps)/3
    frame_count = 0


    target_width = 512
    target_height = 512

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if frame_count % frame_jump == 0:
            resized_frame = cv2.resize(frame, (target_width, target_height))
            _, buffer = cv2.imencode(".jpg", resized_frame)
            videoframe.append(base64.b64encode(buffer).decode("utf-8"))
        frame_count += 1
        
    video.release()
    print(len(videoframe), "frames read.")
    return videoframe

client = OpenAI(api_key='') #your key

system='''
You are given a single grid video <GRID_VIDEO> and basic information of the object.
Layout (2 rows x 3 columns), with zero-based indices (row, col):
Top row:    (0,0)=GT, (0,1)=A, (0,2)=B
Bottom row: (1,0)=empty/ignore, (1,1)=C, (1,2)=D

Task:
1) Slice the grid accordingly and analyze **GT**: determine its category and motion pattern.
2) For **A/B/C/D**, evaluate similarity to GT in motion and geometry (after mentally aligning via rigid transform).
3) Ignore material/texture/color/lighting and pure orientation/viewpoint differences.
4) Rank A/B/C/D by similarity to GT. Do not rely on intra-candidate comparisons as the primary basis.
5) Return the JSON specified in the system prompt.
Json schema:{
    "A": {"geometry_rank": x, "motion_rank": x},
    "B": {"geometry_rank": x, "motion_rank": x}
    "C": {"geometry_rank": x, "motion_rank": x}
    "D": {"geometry_rank": x, "motion_rank": x}
}
'''
prompt="Evaluate the geometry and kinematic reasonableness of the predicted video against the GT video and return exactly one JSON object per the specified schema."
                



save_dir='./evaluation_video'
methodname='results'

os.makedirs(os.path.join(save_dir), exist_ok=True)
os.makedirs(os.path.join(save_dir,methodname), exist_ok=True)

evaluationpath1='./evaluation_video'
basciinfo='./finaljson'   # gt information for better evaluation
filelist=os.listdir(evaluationpath1)

alllist={}

for filename in filelist:
    filename=filename[:-4]
    if os.path.exists(os.path.join(save_dir,methodname,filename+'.json')):
        continue

    jsonfile=os.path.join(basciinfo,filename+'.json')

    with open(jsonfile,'r') as fp:
        jsondata=json.load(fp)

    basicinfo='This is a video of '+jsondata['object_name']+'. '+prompt

    base64Frameseval1=readvideo(evaluationpath1,filename)
    
    PROMPT_MESSAGES = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                   

                    {
                        "type": "text",
                        "text": basicinfo
                    },

                    *[{
                        "type": "image_url",
                        "image_url": {
                            "url": 'data:image/jpeg;base64,' + frame,
                        }
                    } for frame in base64Frameseval1],
                    

                ],
            },
        ]
    

    response = client.chat.completions.create(
        model="gpt-5-chat-latest",
        messages=PROMPT_MESSAGES,
        temperature=0,
        top_p=0,
    )

    with open(os.path.join(save_dir,methodname,filename+'.json'),'w') as file:
        file.write( response.choices[0].message.content)