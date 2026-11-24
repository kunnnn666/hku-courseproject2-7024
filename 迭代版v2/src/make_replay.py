import json, subprocess
from pathlib import Path
from PIL import Image,ImageDraw
import imageio_ffmpeg
run=Path('results/demo_20261101_161336'); s=json.loads((run/'state.json').read_text(encoding='utf-8')); a=s['points']; pts=[p[k] for p in a for k in ('estimated','reference')]; lo=[min(p[i] for p in pts) for i in (0,1)]; hi=[max(p[i] for p in pts) for i in (0,1)];scale=min(560/max(.01,hi[0]-lo[0]),400/max(.01,hi[1]-lo[1]))
def xy(p):return (680+(p[0]-lo[0])*scale,470-(p[1]-lo[1])*scale)
out=Path('results/step234_replay_100frames.mp4'); assert not out.exists()
cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x560','-r','10','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=open('results/video_encoder.log','w'))
for i in range(len(a)):
 im=Image.new('RGB',(1280,560),'#f4f8f8'); draw=ImageDraw.Draw(im)
 im.paste(Image.open(run/'frames'/f'{i:04d}_rgb.jpg'),(10,55)); im.paste(Image.open(run/'frames'/f'{i:04d}_depth.jpg'),(340,55)); draw.text((10,16),f'RGB-D odometry | frame {i+1}/{len(a)} | timestamp {a[i]["timestamp"]:.6f}',fill='#143f46');draw.text((10,310),'RGB input',fill='#143f46');draw.text((340,310),'Depth preview: black = missing',fill='#143f46');draw.text((680,16),'XY trajectory: blue estimate / orange reference',fill='#143f46');draw.text((10,510),'Playback 10 fps; not algorithm processing speed. SE(3) alignment; no scale.',fill='#143f46')
 if i>0:
  draw.line([xy(x['reference']) for x in a[:i+1]],fill='#e78a3b',width=3);draw.line([xy(x['estimated']) for x in a[:i+1]],fill='#2778bb',width=3)
 p.stdin.write(im.tobytes())
p.stdin.close();rc=p.wait();assert rc==0;print(f'PASS video: {out}, frames={len(a)}, fps=10; source {run}')
