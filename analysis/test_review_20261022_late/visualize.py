"""Read-only contact sheets and timed reference comparisons from recordings."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).parent

def read(p): return [json.loads(x) for x in p.open()]
for tag,seconds in [('104416',[3,8,12,16,20,24,28,32]),('104534',[3,7,11,15,19,23,27,30]),('104728',[3,6,9,12,15,18,21,24]),('105113',[3,5,7,9,11])]:
 p=next(Path('deployment_runs').glob('20261022T'+tag+'*')); tr=read(p/'trajectory.jsonl');t0=tr[0]['monotonic_timestamp']; stamps=read(p/'videos/timestamps.jsonl')
 sheet=Image.new('RGB',(424*3,264*len(seconds)),'white');draw=ImageDraw.Draw(sheet)
 for col,cam in enumerate(['cam_high','cam_left_wrist','cam_right_wrist']):
  camst=[x for x in stamps if x['camera']==cam];cap=cv2.VideoCapture(str(p/'videos'/f'{cam}.mp4'))
  for row,sec in enumerate(seconds):
   st=min(camst,key=lambda x:abs(x['monotonic_timestamp']-t0-sec));cap.set(cv2.CAP_PROP_POS_FRAMES,st['frame_index']);ok,frame=cap.read()
   if ok:sheet.paste(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((424,240)),(424*col,264*row+24))
   draw.text((424*col+4,264*row+4),f'{tag} +{sec}s {cam}',fill='black')
  cap.release()
 sheet.save(OUT/f'frames_{tag}.jpg')
for tag,j,window in [('104416',8,(14.8,17.5)),('104728',8,(8,10.6)),('104306',8,(28.5,31.8))]:
 p=next(Path('deployment_runs').glob('20261022T'+tag+'*'));tr=read(p/'trajectory.jsonl');t0=tr[0]['monotonic_timestamp'];ms=read(p/'model_commands.jsonl');sv=read(p/'servo_commands.jsonl')
 fig,ax=plt.subplots(figsize=(11,5));count=0
 for m in ms:
  cap=m['captured_monotonic']-t0;arr=m['arrived_monotonic']-t0
  if not m.get('accepted') or not window[0]-.7<cap<window[1]-.2:continue
  a=np.load(p/m['action_file'])['raw_actions'];t=cap+(np.arange(len(a))+1)/20
  ix=(t>=window[0])&(t<=window[1]);ax.plot(t[ix],a[ix,j],'.--',lw=.8,alpha=.6,label=f'raw g{m["generation"]} return {arr:.2f}s');count+=1
 s=[x for x in sv if window[0] <= max(v['joint_ctrl_monotonic'] for v in x['sides'].values())-t0<=window[1]]
 t=[max(v['joint_ctrl_monotonic'] for v in x['sides'].values())-t0 for x in s]
 ax.plot(t,[x['command_qpos'][j] for x in s],color='black',lw=2.5,label='published command')
 ax.plot([x['feedback_read_monotonic']-t0 for x in s],[x['feedback_qpos'][j] for x in s],color='red',lw=1.5,label='measured feedback')
 ax.set(xlim=window,xlabel='Seconds from recording start',ylabel='Right joint 2 (rad)',title=f'{tag}: absolute raw chunks at original timestamps vs command/feedback')
 ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(OUT/f'reversal_{tag}.png',dpi=140);plt.close(fig)
