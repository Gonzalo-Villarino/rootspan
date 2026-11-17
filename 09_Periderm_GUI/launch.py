"""Double-click launcher for the user; no terminal commands to remember."""
from pathlib import Path
import subprocess,sys,time,urllib.request,webbrowser
from approved_release import verify_release
release=verify_release()
url='http://127.0.0.1:8766'
process=None
def ready():
    try:
        with urllib.request.urlopen(url+'/api/state',timeout=1) as r:
            import json
            state=json.load(r)
            return (state.get('local_only') is True and
                    (state.get('approved_release') or {}).get('manifest_sha256')==release['manifest_sha256'])
    except Exception:return False
if not ready():
    process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('server.py'))],cwd=Path(__file__).parent)
    for _ in range(60):
        if ready():break
        if process.poll() is not None:raise SystemExit('The local app could not start. Port8766 may be in use.')
        time.sleep(.5)
if ready():
    webbrowser.open(url)
    print('RootSpan is open in your browser. Keep this launcher window open while analysing.')
    if process is not None:
        try:process.wait()
        except KeyboardInterrupt:process.terminate()
else:raise SystemExit('The app is taking longer than expected to start.')
