from playwright.sync_api import sync_playwright
from pathlib import Path
import json,time
out=Path('results/browser_validation.json'); evidence={'external_requests':[],'steps':[],'errors':[]}
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True,args=['--no-proxy-server'])
 page=browser.new_page(viewport={'width':1400,'height':1000})
 def route(r):
  u=r.request.url
  if u.startswith('http://127.0.0.1:8765/'):r.continue_()
  else:evidence['external_requests'].append(u);r.abort()
 page.route('**/*',route);page.on('pageerror',lambda e:evidence['errors'].append(str(e)))
 page.goto('http://127.0.0.1:8765/demo/index.html');page.wait_for_function("document.querySelector('#runs').options.length>0")
 page.select_option('#runs','step2_short_v2');page.wait_for_function("document.querySelector('#status').textContent.includes('已完成')")
 assert page.locator('#rgb').evaluate('(e)=>e.complete && e.naturalWidth>0'); evidence['steps'].append('saved result loaded; RGB image decoded')
 page.locator('#seek').fill('4');assert '第 5 /' in page.locator('#frame').inner_text();page.locator('#showgt').uncheck();page.locator('#showgt').check(); evidence['steps'].append('seek and reference visibility operated')
 page.locator('#play').click();page.wait_for_timeout(600);assert page.locator('#play').inner_text()=='暂停回放';page.locator('#play').click();evidence['steps'].append('play and pause operated')
 page.locator('#start').click();page.wait_for_function("document.querySelector('#runs').value.startsWith('demo_')",timeout=30000);page.wait_for_function("document.querySelector('#status').textContent.includes('运行中')",timeout=30000); first=page.locator('#status').inner_text(); page.wait_for_timeout(3500);second=page.locator('#status').inner_text();assert first!=second,(first,second);evidence['steps'].append('live state advanced: '+first+' -> '+second)
 page.locator('#stop').click();page.wait_for_function("document.querySelector('#status').textContent.includes('已停止')",timeout=30000);evidence['steps'].append('stop processed and final result available: '+page.locator('#status').inner_text());page.screenshot(path='results/demo_validation.png',full_page=True);browser.close()
out.write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(evidence,ensure_ascii=False,indent=2))
