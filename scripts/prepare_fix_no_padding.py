"""Make a new dataset, removing verified baked-in letterbox bands; keep all frames/actions."""
import argparse,json,shutil,subprocess,os,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);args=p.parse_args()
 if args.destination.exists():raise RuntimeError('destination already exists; refusing to overwrite')
 videos=sorted(args.source.glob('videos/**/*.mp4'));audit=[]
 cv2.setNumThreads(1)
 # Verify common crop geometry on multiple frames of EVERY video before creating output.
 for path in videos:
  cap=cv2.VideoCapture(str(path));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
  for index in sorted(set([0,n//2,n-1])):
   cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
   if not ok or frame.shape!=(224,224,3):raise RuntimeError(f'invalid video: {path}')
   # The known resize produces 127 content rows, y=48:175. Allow codec bleed next to the edge.
   bands=np.concatenate([frame[:46].reshape(-1,3),frame[178:].reshape(-1,3)])
   if np.percentile(bands,99.9)>12:raise RuntimeError(f'expected black bands not found: {path} frame {index}')
   if np.mean(frame[50:173])<5:raise RuntimeError(f'unusable dark content: {path}')
  cap.release();audit.append(dict(path=str(path.relative_to(args.source)),frames=n,fps=fps))
 print('Verified bands in',len(videos),'videos',flush=True)
 temp=args.destination.with_name(args.destination.name+'.processing');temp.mkdir(parents=True)
 for child in args.source.iterdir():
  if child.name=='videos':continue
  if child.is_dir():shutil.copytree(child,temp/child.name)
  else:shutil.copy2(child,temp/child.name)
 for record in audit:
  source=args.source/record['path'];dest=temp/record['path'];dest.parent.mkdir(parents=True,exist_ok=True)
  cap=cv2.VideoCapture(str(source));count=0
  command=[str(Path(sys.executable).parent/'ffmpeg'),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','224x224','-r',str(record['fps']),'-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(dest)]
  proc=subprocess.Popen(command,stdin=subprocess.PIPE)
  try:
   while True:
    ok,frame=cap.read()
    if not ok:break
    rgb=cv2.cvtColor(frame[48:175,:],cv2.COLOR_BGR2RGB)
    output=np.asarray(Image.fromarray(rgb).resize((224,224),Image.Resampling.BILINEAR))
    proc.stdin.write(output.tobytes());count+=1
  finally:
   cap.release();proc.stdin.close()
  if proc.wait()!=0 or count!=record['frames']:raise RuntimeError(f'encode/frame mismatch: {source}')
  check=cv2.VideoCapture(str(dest));actual=int(check.get(cv2.CAP_PROP_FRAME_COUNT));ok,img=check.read();check.release()
  if actual!=count or not ok:raise RuntimeError(f'output decode mismatch: {dest}')
  if np.mean(img[:20])<5:raise RuntimeError(f'output still contains a black top band: {dest}')
  print(json.dumps(dict(video=record['path'],frames=count)),flush=True)
 # Split stays the same; only change dataset identity. Norm is rebuilt by training job.
 split=temp/'meta/train_test_split.json'
 if split.exists():
  obj=json.loads(split.read_text());obj['dataset_id']=args.destination.name;split.write_text(json.dumps(obj,indent=2)+'\n')
 norm=temp/'meta/openpi_norm_stats.json'
 if norm.exists():norm.unlink()
 manifest=dict(source=str(args.source),destination=str(args.destination),crop_xyxy=[0,48,224,175],resize='PIL bilinear stretch to 224x224',videos=audit,
  note='Only known padding bands removed. Source image detail lost during original downsampling cannot be recovered. Actions, timestamps, episode split retained.')
 (temp/'meta/image_preprocessing.json').write_text(json.dumps(manifest,indent=2)+'\n')
 os.rename(temp,args.destination);print('COMPLETED',args.destination,flush=True)
if __name__=='__main__':main()
