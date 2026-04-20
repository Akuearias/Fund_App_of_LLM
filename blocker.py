import psutil
import time
import threading

running = False
blocked_apps = []

def kill_apps():
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            name = proc.info['name']
            if name:
                for app in blocked_apps:
                    if app.lower() in name.lower():
                        proc.kill()
        except:
            pass

def loop():
    while running:
        kill_apps()
        time.sleep(2)

def start(apps):
    global running, blocked_apps
    blocked_apps = apps
    running = True
    threading.Thread(target=loop, daemon=True).start()

def stop():
    global running
    running = False
