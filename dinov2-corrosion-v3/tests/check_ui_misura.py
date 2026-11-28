"""Controllo dell'interfaccia «📐 Misura» con Edge headless (CDP), sul server v3 avviato qui sulla porta 8799.

Non tocca il server dell'utente (8766 / 8767). Uso:
    .venv\\Scripts\\python.exe tests\\check_ui_misura.py [--sessione] [--shots CARTELLA]
Esce con codice 0 se tutti i controlli passano. Con --sessione prova anche la finestra multi-foto (task 10).
"""
import asyncio
import base64
import json
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

ROOT = pathlib.Path(__file__).resolve().parents[1]
FOTO = pathlib.Path(r"C:\Users\gdecandia\Desktop\Oggi2\tralicci corrosione")
PHOTO = FOTO / (sys.argv[sys.argv.index("--foto") + 1] if "--foto" in sys.argv else "Camera_4_2026-07-14_11.01.03.000.jpg")
EXTRA = None  # creata in run(): Camera_4 con il drone 10 m più in alto (stesso sostegno, solo per provare l'interfaccia)


def foto_spostata(cartella: pathlib.Path) -> pathlib.Path:
    data = PHOTO.read_bytes()
    assert data.count(b"+781.875") == 1, "AbsoluteAltitude di Camera_4 non trovata"
    out = cartella / "Camera_4_quota_piu_10m.jpg"
    out.write_bytes(data.replace(b"+781.875", b"+791.875"))  # stessa lunghezza: JPEG intatto
    return out
MODELS = r"C:\01 Tools Brasile\01 -YoloTron\models_auto"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT, W, H = 8799, 1600, 900
SESSIONE = "--sessione" in sys.argv
SHOTS = pathlib.Path(sys.argv[sys.argv.index("--shots") + 1]) if "--shots" in sys.argv else pathlib.Path(tempfile.gettempdir())
FALLITI = []


def check(nome, ok, dettaglio=""):
    print(("OK   " if ok else "FAIL ") + nome + (f"  [{dettaglio}]" if dettaglio and not ok else ""))
    if not ok:
        FALLITI.append(nome)


async def run():
    misure_dir = tempfile.mkdtemp(prefix="v3_misure_")  # sessioni di prova fuori dalla cartella reale
    srv = subprocess.Popen([str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "uvicorn", "main:app", "--host",
                            "127.0.0.1", "--port", str(PORT)], cwd=ROOT, stdout=subprocess.DEVNULL,
                           env={**__import__("os").environ, "V3_MISURE_DIR": misure_dir},
                           stderr=open(SHOTS / "check_ui_server.err", "w"))
    edge = None
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/status", timeout=2); break
            except Exception:
                time.sleep(.5)
        edge = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", "--remote-debugging-port=9334",
                                 f"--user-data-dir={SHOTS / 'edge_check_ui'}", f"--window-size={W},{H}", "about:blank"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9334/json")); break
            except Exception:
                time.sleep(.3)
        ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
        async with websockets.connect(ws_url, max_size=80_000_000) as ws:
            n = 0

            async def cmd(method, **params):
                nonlocal n
                n += 1
                my = n
                await ws.send(json.dumps({"id": my, "method": method, "params": params}))
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("id") == my:
                        if "error" in m:
                            raise RuntimeError(m["error"])
                        return m.get("result", {})

            async def js(expr):
                r = await cmd("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
                if r.get("exceptionDetails"):
                    raise RuntimeError(r["exceptionDetails"])
                return r["result"].get("value")

            async def shot(name):
                r = await cmd("Page.captureScreenshot", format="png")
                (SHOTS / f"misura_{PHOTO.stem[:8]}_{name}.png").write_bytes(base64.b64decode(r["data"]))
                print("screenshot", SHOTS / f"misura_{PHOTO.stem[:8]}_{name}.png")

            async def attendi(expr, secondi=120):
                for _ in range(secondi * 2):
                    if await js(expr):
                        return True
                    await asyncio.sleep(.5)
                return False

            async def clic(x, y, volte=1):
                for k in range(1, volte + 1):
                    for t in ("mousePressed", "mouseReleased"):
                        await cmd("Input.dispatchMouseEvent", type=t, x=x, y=y, button="left", clickCount=k)
                await asyncio.sleep(.15)

            async def centro(sel, fx=.5, fy=.5):
                return await js(f"(()=>{{let r=document.querySelector({json.dumps(sel)}).getBoundingClientRect();"
                                f"return [r.left+r.width*{fx},r.top+r.height*{fy}]}})()")

            await cmd("Page.enable"); await cmd("Runtime.enable")
            await cmd("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
            await cmd("Page.navigate", url=f"http://127.0.0.1:{PORT}/"); await asyncio.sleep(2)
            act = [{"id": MODELS + r"\Sostegni _ Targhette V1s.pt"},
                   {"id": MODELS + r"\IsolatoriV2s.pt", "modes": {"Isolatori": "exclude"}}]
            await js("localStorage.setItem('airvision-v2.activeModels',%s)" % json.dumps(json.dumps(act)))
            await cmd("Page.reload"); await asyncio.sleep(4)
            doc = await cmd("DOM.getDocument")
            node = await cmd("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="#photo")
            await cmd("DOM.setFileInputFiles", files=[str(PHOTO)], nodeId=node["nodeId"])
            await attendi("(t=>t.startsWith('Trovati')||t.includes('Nessun oggetto'))(document.getElementById('objectStatus').textContent)", 300)
            await js("document.getElementById('analyze').click()")
            ok = await attendi("document.getElementById('output').style.display==='block'||!!document.getElementById('error').textContent.trim()", 1200)
            err = await js("document.getElementById('error').textContent.trim()")
            check("analisi completata", ok and not err, err)
            await js("window.__urls=[];const _f=window.fetch;window.fetch=(u,o)=>{window.__urls.push(String(u));return _f(u,o)}")
            await js("document.getElementById('mapbox').click()"); await asyncio.sleep(1.5)
            check("pulsante Misura presente e spento",
                  await js("!!document.getElementById('vMeasure')&&!document.getElementById('vMeasure').classList.contains('on')"))
            check("pannello Misura nascosto all'apertura", await js("!!document.getElementById('measurePanel')&&document.getElementById('measurePanel').hidden"))
            await js("document.getElementById('vMeasure').click()")
            ok = await attendi("(t=>t.includes('Drone')&&t.includes('STIMA'))(document.getElementById('mpPhoto').textContent)", 90)
            check("pannello mostra drone e STIMA", ok, await js("document.getElementById('mpPhoto').textContent"))
            check("misure automatiche in m²", await js("document.getElementById('mpAuto').textContent.includes('m²')"))
            print("PANNELLO:", await js("[...document.querySelectorAll('#mpPhotoBody .mkv, #mpAutoList .mrow, #mpAutoList .msub')].map(e=>e.textContent.replace(/\s+/g,' ')).join(' | ')"))
            await shot("1_pannello")
            await js("document.getElementById('mpLine').click()")
            x0, y0 = await centro("#vbody .vpane.map .vstage", .45, .30)
            x1, y1 = await centro("#vbody .vpane.map .vstage", .45, .55)
            await clic(x0, y0); await clic(x1, y1); await clic(x1, y1, 2)
            ok = await attendi("document.getElementById('mpList').textContent.includes('±')", 60)
            check("linea misurata con ±", ok, await js("document.getElementById('mpList').textContent"))
            await shot("2_linea")
            await js("document.querySelector('#mpList .mp-pdf').click()")
            await js("document.getElementById('pdf').click()"); await asyncio.sleep(3)
            urls = await js("window.__urls")
            check("PDF rigenerato con le misure", any("/api/misura/pdf" in u for u in urls), str(urls))
            if SESSIONE:
                await js("document.getElementById('mpOpenSession').click()")
                ok = await attendi("document.querySelectorAll('#sdStrip .sd-thumb').length===1", 30)
                check("sessione: 1 miniatura principale", ok and await js("document.querySelector('#sdStrip .sd-thumb').textContent.includes('★')"))
                doc = await cmd("DOM.getDocument")
                node = await cmd("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="#sdAdd")
                await cmd("DOM.setFileInputFiles", files=[str(foto_spostata(SHOTS))], nodeId=node["nodeId"])
                check("sessione: foto aggiunta", await attendi("document.querySelectorAll('#sdStrip .sd-thumb').length===2", 60))
                await js("document.querySelectorAll('#sdStrip .sd-thumb')[1].click()"); await asyncio.sleep(2)
                xa, ya = await centro("#sdLeft .sd-stage", .5, .8)
                await clic(xa, ya)
                ok = await attendi("!!document.querySelector('#sdRight svg path')", 30)
                check("sessione: curva suggerita a destra", ok)
                await shot("3_sessione")
                await js("document.getElementById('sdCalc').click()")
                ok = await attendi("document.getElementById('sdChecks').textContent.includes('INCOMPLETA')", 60)
                check("sessione: INCOMPLETA con pochi punti", ok and await js("/T\\d/.test(document.getElementById('sdChecks').textContent)"),
                      await js("document.getElementById('sdChecks').textContent"))
                await js("document.getElementById('sdClose').click()"); await asyncio.sleep(.5)
                await js("document.getElementById('mpOpenSession').click()")
                ok = await attendi("document.getElementById('sdTies').textContent.includes('T1')", 30)
                check("sessione: T1 ancora presente dopo la riapertura", ok)
    finally:
        if edge:
            edge.terminate()
        srv.terminate()
    print("FALLITI:", FALLITI if FALLITI else "nessuno")
    return 1 if FALLITI else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
