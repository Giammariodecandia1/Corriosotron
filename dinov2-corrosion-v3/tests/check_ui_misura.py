"""Controllo dell'interfaccia «📐 Misura» con Edge headless (CDP), sul server v3 avviato qui sulla porta 8799.

Non tocca il server dell'utente (8766 / 8767). Caso reale: traliccio 23341A1-074 (Codrongianos) fotografato da ovest
(A, foto analizzata) e da est (B, aggiunta nella sessione); i 4 tie point (cimino e punte delle 3 mensole) si
cliccano con il mouse in entrambe le foto. Uso:
    .venv\\Scripts\\python.exe tests\\check_ui_misura.py [--shots CARTELLA]
Esce con codice 0 se tutti i controlli passano.
"""
import asyncio
import base64
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRALICCIO = pathlib.Path(r"\\192.168.199.10\Elaborazioni\20_SVILUPPO\00_caricamenti\drone\Terna\Terna Termografia Tralicci"
                         r"\Sardegna\Codrongianos\20231130\23341A1-074\09_traliccio")
FOTO_A = TRALICCIO / "01_tronco_lato_a" / "DJI_20231130151032_0022_W.JPG"
FOTO_B = TRALICCIO / "02_tronco_lato_b" / "DJI_20231130151056_0025_W.JPG"
NATIVA = (4056, 3040)
# tie point in pixel della foto originale: (A, B)
TIE = {"T1": ((2021, 1247), (1988, 1238)), "T2": ((1812, 1524), (2160, 1590)),
       "T3": ((2268, 1582), (1833, 1371)), "T4": ((2280, 1880), (1830, 1592))}
MODELS = r"C:\01 Tools Brasile\01 -YoloTron\models_auto"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT, W, H = 8799, 1600, 900
SHOTS = pathlib.Path(sys.argv[sys.argv.index("--shots") + 1]) if "--shots" in sys.argv else pathlib.Path(tempfile.gettempdir())
FALLITI = []


def check(nome, ok, dettaglio=""):
    print(("OK   " if ok else "FAIL ") + nome + (f"  [{dettaglio}]" if dettaglio and not ok else ""))
    if not ok:
        FALLITI.append(nome)


async def run():
    if not FOTO_A.exists():
        print("cartella di rete Codrongianos non raggiungibile:", FOTO_A)
        return 2
    locale = pathlib.Path(tempfile.mkdtemp(prefix="v3_foto_"))
    PHOTO, EXTRA = (pathlib.Path(shutil.copy(f, locale / f"{lato}_{f.name}")) for lato, f in (("A", FOTO_A), ("B", FOTO_B)))
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
            await js("document.getElementById('vMeasure').click()"); await asyncio.sleep(1)
            testo = await js("document.getElementById('measurePanel').textContent")
            check("pannello: invito alla sessione, nessuna stima da una foto",
                  "solo da più foto" in testo and "Disponibili dopo" in testo and "Drone" not in testo, testo)
            check("nessuna chiamata a /api/misura/stima", not any("/stima" in u for u in await js("window.__urls")))
            await js("document.getElementById('mpLine').click()")
            x0, y0 = await centro("#vbody .vpane.map .vstage", .45, .30)
            x1, y1 = await centro("#vbody .vpane.map .vstage", .45, .55)
            await clic(x0, y0); await clic(x1, y1); await clic(x1, y1, 2)
            check("linea disegnata in attesa della sessione",
                  await attendi("(t=>t.includes('L1')&&t.includes('—'))(document.getElementById('mpList').textContent)", 10),
                  await js("document.getElementById('mpList').textContent"))
            await shot("1_pannello_senza_sessione")

            await js("document.getElementById('mpOpenSession').click()")
            ok = await attendi("document.querySelectorAll('#sdStrip .sd-thumb').length===1", 30)
            check("sessione: 1 miniatura principale", ok and await js("document.querySelector('#sdStrip .sd-thumb').textContent.includes('★')"))
            doc = await cmd("DOM.getDocument")
            node = await cmd("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="#sdAdd")
            await cmd("DOM.setFileInputFiles", files=[str(EXTRA)], nodeId=node["nodeId"])
            check("sessione: foto B aggiunta e misurabile", await attendi(
                "document.querySelectorAll('#sdStrip .sd-thumb').length===2&&!document.querySelector('#sdStrip .bad')", 60))
            await asyncio.sleep(2)
            for pid, ((ua, va), (ub, vb)) in TIE.items():  # posizioni come frazione della foto: valgono a ogni risoluzione
                await js(f"document.querySelector('#sdTies .sd-tie[data-pid=\"{pid}\"]').click()"); await asyncio.sleep(.3)
                await clic(*await centro("#sdLeft .sd-stage", ua / NATIVA[0], va / NATIVA[1]))
                if pid == "T1":
                    check("sessione: curva suggerita a destra", await attendi("!!document.querySelector('#sdRight svg path')", 30))
                    await shot("2_sessione_curva")
                await clic(*await centro("#sdRight .sd-stage", ub / NATIVA[0], vb / NATIVA[1]))
                await asyncio.sleep(.5)
            n = await js("[...document.querySelectorAll('#sdTies .sd-tie')].filter(r=>r.textContent.includes('2/2 foto')).length")
            check("sessione: 4 tie point in 2 foto", n == 4, await js("document.getElementById('sdTies').textContent"))
            await js("document.getElementById('sdCalc').click()")
            ok = await attendi("document.getElementById('sdChecks').textContent.includes('NON VALIDATA')", 120)
            checks = await js("document.getElementById('sdChecks').textContent")
            check("sessione: calcolo completo, NON VALIDATA (niente RTK)", ok and "Riproiezione" in checks, checks)
            print("CONTROLLI:", checks)
            await shot("3_sessione_calcolata")
            await js("document.getElementById('sdClose').click()"); await asyncio.sleep(1)
            check("pannello: misure automatiche in m²", await attendi("document.getElementById('mpAuto').textContent.includes('m²')", 30))
            check("pannello: linea L1 misurata con ±",
                  await attendi("document.getElementById('mpList').textContent.includes('±')", 60),
                  await js("document.getElementById('mpList').textContent"))
            print("PANNELLO:", await js("[...document.querySelectorAll('#mpPhotoBody .mkv, #mpAutoList .mrow, #mpList .mrow')].map(e=>e.textContent.replace(/\\s+/g,' ')).join(' | ')"))
            await shot("4_pannello_misure")
            await js("document.querySelector('#mpList .mp-pdf').click()")
            await js("document.getElementById('pdf').click()"); await asyncio.sleep(3)
            urls = await js("window.__urls")
            check("PDF rigenerato con le misure", any("/api/misura/pdf" in u for u in urls), str(urls))
            await js("document.getElementById('mpOpenSession').click()")
            ok = await attendi("document.getElementById('sdTies').textContent.includes('T4 · 2/2')", 30)
            check("sessione: tie point presenti dopo la riapertura", ok, await js("document.getElementById('sdTies').textContent"))
    finally:
        if edge:
            edge.terminate()
        srv.terminate()
    print("FALLITI:", FALLITI if FALLITI else "nessuno")
    return 1 if FALLITI else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
