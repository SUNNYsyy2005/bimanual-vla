"""Export one held-out observation for hardware-free local policy verification."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
episode,frame=7,40
split=json.loads((a.dataset/'meta/train_test_split.json').read_text());assert episode in split['test_episodes']
t=pq.read_table(a.dataset/f'data/chunk-000/episode_{episode:06d}.parquet')
tasks={r['task_index']:r['task'] for r in map(json.loads,(a.dataset/'meta/tasks.jsonl').read_text().splitlines())}
items=dict(state=np.asarray(t['observation.state'][frame].as_py(),np.float32),truth=np.asarray(t['action'].to_pylist()[frame:frame+50],np.float32),prompt=np.array(tasks[t['task_index'][frame].as_py()]),episode=episode,frame=frame)
for camera in ('cam_high','cam_right_wrist'):
 c=cv2.VideoCapture(str(a.dataset/f'videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4'));c.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=c.read();c.release();assert ok
 items[camera]=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
np.savez_compressed(a.output,**items)
print('saved heldout episode',episode,'frame',frame)
