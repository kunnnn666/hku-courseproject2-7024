from playwright.sync_api import sync_playwright
import json
with sync_playwright() as p:
 b=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True); page=b.new_page(viewport={'width':1400,'height':1000}); external=[]
 def route(r):
  if r.request.url.startswith('http://127.0.0.1:8765/'):r.continue_()
  else:external.append(r.request.url);r.abort()
 page.route('**/*',route);page.goto('http://127.0.0.1:8765/demo/index.html');page.wait_for_function("document.querySelector('#status').textContent.includes('已完成')",timeout=30000);assert page.locator('#runs').input_value()=='step234_full_retry';assert '792' in page.locator('#stats').inner_text() and '790' in page.locator('#metrics').inner_text();assert page.locator('#rgb').evaluate('(e)=>e.complete&&e.naturalWidth>0');page.screenshot(path='results/demo_full_validation.png',full_page=True);print(json.dumps({'default_full_result':True,'stats_792':True,'associated_790':True,'external_requests':external},ensure_ascii=False));b.close()
