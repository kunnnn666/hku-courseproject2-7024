import http.server, json, os, subprocess, sys, time, urllib.parse, webbrowser
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RESULTS=ROOT/'results'; process=None; current=None
class Handler(http.server.SimpleHTTPRequestHandler):
 def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT),**kw)
 def send_video(self,url_path):
  # Browsers need byte ranges to seek within longer local MP4 files.
  path=Path(self.translate_path(url_path)).resolve()
  if not path.is_relative_to(ROOT.resolve()) or not path.is_file():return self.send_error(404)
  size=path.stat().st_size; start=0; end=size-1; requested=self.headers.get('Range')
  if requested:
   try:
    if not requested.startswith('bytes=') or ',' in requested:raise ValueError('single byte range required')
    left,right=requested[6:].split('-',1)
    if left:
     start=int(left); end=min(int(right),size-1) if right else size-1
    else:
     suffix=int(right)
     if suffix<=0:raise ValueError('invalid suffix')
     start=max(0,size-suffix)
    if start<0 or start>=size or end<start:raise ValueError('unsatisfiable')
   except (ValueError,TypeError):
    self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return
  self.send_response(206 if requested else 200)
  self.send_header('Content-Type','video/mp4');self.send_header('Accept-Ranges','bytes')
  self.send_header('Content-Length',str(max(0,end-start+1)))
  if requested:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
  self.end_headers()
  try:
   with path.open('rb') as f:
    f.seek(start);remaining=end-start+1
    while remaining>0:
     block=f.read(min(1024*1024,remaining))
     if not block:break
     self.wfile.write(block);remaining-=len(block)
  except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
 def send_json(self,v,code=200):
  data=json.dumps(v,ensure_ascii=False).encode(); self.send_response(code); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(data)
 def do_GET(self):
  u=urllib.parse.urlsplit(self.path)
  if u.path=='/api/runs':
   runs=sorted([p.name for p in RESULTS.iterdir() if p.is_dir() and (p/'state.json').exists()],reverse=True); return self.send_json({'runs':runs,'active':current if process and process.poll() is None else None})
  if u.path=='/api/state':
   name=urllib.parse.parse_qs(u.query).get('run',[''])[0]
   if not name or Path(name).name!=name:return self.send_json({'error':'invalid run'},400)
   try:
    state=json.loads((RESULTS/name/'state.json').read_text(encoding='utf-8'))
    if name==current and process and process.poll() is not None and state.get('status')=='running': state['status']='error'; state['reason']='处理进程异常退出，请查看对应 console.log'
    return self.send_json(state)
   except Exception as e:return self.send_json({'error':str(e)},404)
  if u.path.lower().endswith('.mp4'):return self.send_video(u.path)
  return super().do_GET()
 def do_POST(self):
  global process,current
  origin=self.headers.get('Origin',''); host=self.headers.get('Host','')
  if origin != 'http://'+host or host.split(':')[0] not in ('127.0.0.1','localhost'):return self.send_json({'error':'local origin required'},403)
  if self.path=='/api/start':
   if process and process.poll() is None:return self.send_json({'error':'已有任务运行'},409)
   name='demo_'+time.strftime('%Y%m%d_%H%M%S'); out=RESULTS/name
   env=os.environ.copy(); env['OMP_NUM_THREADS']='4'; log=(RESULTS/(name+'_console.log')).open('w',encoding='utf-8'); process=subprocess.Popen([sys.executable,str(ROOT/'src'/'continuous_vo.py'),'--output',str(out),'--frames','100'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)); log.close(); current=name; return self.send_json({'run':name})
  if self.path=='/api/stop':
   if not process or process.poll() is not None:return self.send_json({'error':'没有运行中的任务'},409)
   (RESULTS/current/'STOP').touch(); return self.send_json({'message':'已请求停止；当前帧处理结束后生效'})
  self.send_json({'error':'unknown'},404)
if __name__=='__main__':
 server=http.server.ThreadingHTTPServer(('127.0.0.1',8765),Handler); print('Local only: http://127.0.0.1:8765/demo/index.html',flush=True)
 if '--no-browser' not in sys.argv:webbrowser.open('http://127.0.0.1:8765/demo/index.html')
 try:server.serve_forever()
 except KeyboardInterrupt:print('Server stopped; an active processing task continues until finished or stopped from the page.')
