
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import torch
from collections import deque
import sys
import os

# Try importing pynput for keyboard control
try:
    from pynput.keyboard import Key, Controller
    params_keyboard = True
    keyboard = Controller()
except ImportError:
    params_keyboard = False
    keyboard = None
    print("WARNING: 'pynput' library not found. Keyboard simulation disabled. Install with: pip install pynput")

# Import project modules
# Ensure the current directory is in sys.path to find 'training' module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.features.logmel import LogMelExtractor
from training.models.cnn_kws import SmallCNNKWS

# --- Constants & Configuration ---
LABELS = ["up", "down", "left", "right", "silence", "noise"]
COMMANDS = ["up", "down", "left", "right"]
SR = 16000
WIN_SEC = 1.0
HOP_SEC = 0.15
WIN = int(SR * WIN_SEC)
BLOCK = int(SR * HOP_SEC)

# Default Thresholds (from infer_mic.py)
THRESH_CMD = {
    "up": 0.88,
    "down": 0.88,
    "right": 0.88,
    "left": 0.70, 
}
DEFAULT_CMD_THRESH = 0.88

SMOOTH_K = 3
NEED = 2
COOLDOWN_SEC = 0.45
RESET_FRAMES = 4

# Key Mappings (Command -> pynput Key)
# Define default mappings only if keyboard is available, otherwise use strings or placeholders
if params_keyboard:
    KEY_MAPPING = {
        "up": Key.up,
        "down": Key.down,
        "left": Key.left,
        "right": Key.right
    }
else:
    KEY_MAPPING = {
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right"
    }

class VoiceCommandApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Command Simulation")
        self.root.geometry("600x500")
        self.root.resizable(True, True)

        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', font=('Helvetica', 12, 'bold'))
        self.style.configure('TLabel', font=('Helvetica', 14))
        self.style.configure('Status.TLabel', font=('Helvetica', 18, 'bold'), foreground='blue')

        # State vars
        self.is_listening = False
        self.audio_thread = None
        self.stop_event = threading.Event()
        self.msg_queue = queue.Queue()
        
        # Model placeholders
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.feat = None
        
        # UI Setup
        self.setup_ui()
        
        # Load Model (Async to not freeze UI on startup, or just do it here quickly)
        self.load_model()
        
        # Start consumer loop
        self.root.after(100, self.process_queue)


    def setup_ui(self):
        # Top Frame: Controls
        control_frame = ttk.Frame(self.root, padding=20)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_toggle = ttk.Button(control_frame, text="Start Listening", command=self.toggle_listening)
        self.btn_toggle.pack(side=tk.LEFT, padx=10)

        self.btn_settings = ttk.Button(control_frame, text="Key Bindings", command=self.open_settings)
        self.btn_settings.pack(side=tk.RIGHT, padx=10)

        # Middle Frame: Visualization (Arrows)
        viz_frame = ttk.Frame(self.root, padding=20)
        viz_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        
        # Grid layout for arrows
        #       UP
        # LEFT      RIGHT
        #      DOWN
        
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.columnconfigure(1, weight=1)
        viz_frame.columnconfigure(2, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        viz_frame.rowconfigure(1, weight=1)
        viz_frame.rowconfigure(2, weight=1)

        self.lbl_up = tk.Label(viz_frame, text="UP", width=10, height=3, bg="lightgray", font=('Arial', 16, 'bold'))
        self.lbl_up.grid(row=0, column=1, pady=10)

        self.lbl_left = tk.Label(viz_frame, text="LEFT", width=10, height=3, bg="lightgray", font=('Arial', 16, 'bold'))
        self.lbl_left.grid(row=1, column=0, padx=10)

        self.lbl_right = tk.Label(viz_frame, text="RIGHT", width=10, height=3, bg="lightgray", font=('Arial', 16, 'bold'))
        self.lbl_right.grid(row=1, column=2, padx=10)

        self.lbl_down = tk.Label(viz_frame, text="DOWN", width=10, height=3, bg="lightgray", font=('Arial', 16, 'bold'))
        self.lbl_down.grid(row=2, column=1, pady=10)

        self.arrows = {
            "up": self.lbl_up,
            "down": self.lbl_down,
            "left": self.lbl_left,
            "right": self.lbl_right
        }

        # Original colors
        self.default_bg = "lightgray"
        self.active_bg = "#32CD32" # Lime Green

        # Bottom Frame: Status
        status_frame = ttk.Frame(self.root, padding=20)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        self.lbl_status = ttk.Label(status_frame, text="IDLE", style='Status.TLabel')
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        self.lbl_confidence = ttk.Label(status_frame, text="Conf: 0.00")
        self.lbl_confidence.pack(side=tk.RIGHT)

        # Device Label
        self.lbl_device = ttk.Label(status_frame, text="Device: ...", font=('Helvetica', 10))
        self.lbl_device.pack(side=tk.RIGHT, padx=20)

    def load_model(self):
        try:
            # Resolve absolute path to checkpoint
            script_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(script_dir, "checkpoints", "kws_cnn_best.pt")
            
            if not os.path.exists(checkpoint_path):
                messagebox.showerror("Error", f"Checkpoint not found at {checkpoint_path}")
                return

            ckpt = torch.load(checkpoint_path, map_location="cpu")
            self.model = SmallCNNKWS(num_classes=len(LABELS))
            self.model.load_state_dict(ckpt["model"])
            self.model.eval().to(self.device)

            self.feat = LogMelExtractor().to(self.device).eval()
            print(f"Model loaded on {self.device}")
            self.lbl_status.config(text="Model Loaded")
            self.lbl_device.config(text=f"Device: {self.device.upper()}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def toggle_listening(self):
        if self.is_listening:
            self.stop_listening()
        else:
            self.start_listening()

    def start_listening(self):
        if not self.model:
            messagebox.showwarning("Warning", "Model not loaded!")
            return
        self.is_listening = True
        self.stop_event.clear()
        self.btn_toggle.config(text="Stop Listening")
        self.lbl_status.config(text="Listening...", foreground="orange")
        
        # Start thread
        self.audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.audio_thread.start()

    def stop_listening(self):
        self.is_listening = False
        self.stop_event.set()
        self.btn_toggle.config(text="Start Listening")
        self.lbl_status.config(text="Stopped", foreground="black")
        self.reset_arrows()

    def audio_loop(self):
        buffer = np.zeros(WIN, dtype=np.float32)
        history = deque(maxlen=SMOOTH_K)
        last_fire = 0.0
        last_best = "none"
        none_streak = 0

        # Callback for sounddevice
        def sd_callback(indata, frames, t, status):
            if status:
                print(status)
            self.msg_queue.put(("audio_data", indata[:, 0].astype(np.float32)))

        try:
            with sd.InputStream(samplerate=SR, channels=1, blocksize=BLOCK, callback=sd_callback):
                while not self.stop_event.is_set():
                    # Process audio from queue if we were doing it completely decoupled, 
                    # but here we need to process it. 
                    # Actually, better pattern: sd callback puts into queue, we read from queue here.
                    # HOWEVER, heavy inference in callback is bad. The 'infer_mic.py' did inference inside callback.
                    # We can do the same if it's fast enough, but to support "Multithreading for smoothness",
                    # let's have the sd callback push chunks to a queue, and THIS thread pop and process.
                    pass
                    # Wait slightly
                    sd.sleep(100)
        except Exception as e:
            self.msg_queue.put(("error", str(e)))

        # Wait, the structure above relies on 'sd_callback' being called by sounddevice thread.
        # But we want to do inference in THIS thread (self.audio_thread) to avoid blocking the audio callback 
        # (though infer_mic.py did it in callback, which is risky but worked).
        # Let's refactor: audio_loop essentially becomes the worker.
        # But Sounddevice creates its own thread.
        # So: SD Callback -> Buffer Queue -> Audio Loop (Inference) -> UI Queue -> UI Main
        
        # For simplicity and sticking to the request "Help make it run multithreaded/smoother":
        # I'll implement the producer-consumer for audio->inference properly.

    # RE-IMPLEMENTING audio_loop properly
    def audio_loop(self):
        # Allow passing audio chunks from SD thread to Inference thread
        audio_queue = queue.Queue()
        
        def sd_callback(indata, frames, t, status):
            if status:
                print(status)
            # Copy data to avoid buffer issues
            audio_queue.put(indata[:, 0].astype(np.float32).copy())

        buffer = np.zeros(WIN, dtype=np.float32)
        history = deque(maxlen=SMOOTH_K)
        last_fire = 0.0
        last_best = "none"
        none_streak = 0

        # Start InputStream
        stream = sd.InputStream(samplerate=SR, channels=1, blocksize=BLOCK, callback=sd_callback)
        stream.start()

        try:
            while not self.stop_event.is_set():
                try:
                    # Get new data
                    x = audio_queue.get(timeout=1.0) # Wait for audio
                except queue.Empty:
                    continue

                if x.size == 0: continue

                # Rolling buffer
                buffer = np.roll(buffer, -len(x))
                buffer[-len(x):] = x

                wav = torch.from_numpy(buffer.copy())

                # Inference
                with torch.no_grad():
                    logmel = self.feat(wav.to(self.device))
                    inp = logmel.unsqueeze(0)
                    logits = self.model(inp)
                    prob = torch.softmax(logits, dim=-1)[0]

                    top2 = torch.topk(prob, k=2)
                    p1, i1 = float(top2.values[0].item()), int(top2.indices[0].item())
                    p2, i2 = float(top2.values[1].item()), int(top2.indices[1].item())
                    l1, l2 = LABELS[i1], LABELS[i2]

                    # Rule check
                    if l1 == "right" and l2 == "left" and (p1 - p2) < 0.06:
                        pred_label = "left"
                        conf = p2
                    else:
                        pred_label = l1
                        conf = p1

                # Send raw status for "Silence/Noise" visualization
                # If pred_label is silence or noise, show that
                # We send this every frame for "Real time status"
                self.msg_queue.put(("status_update", (pred_label, conf)))

                # Command Logic
                if pred_label in COMMANDS:
                    thr = THRESH_CMD.get(pred_label, DEFAULT_CMD_THRESH)
                    label = pred_label if conf >= thr else "none"
                else:
                    label = "none"

                history.append(label)

                best = "none"
                for cmd in COMMANDS:
                    if history.count(cmd) >= NEED:
                        best = cmd
                        break
                
                now = time.time()
                if best == "none":
                    none_streak += 1
                    if none_streak >= RESET_FRAMES:
                        last_best = "none"
                else:
                    none_streak = 0

                # Trigger Command
                if best != last_best and (now - last_fire) > COOLDOWN_SEC:
                    self.msg_queue.put(("command", (best, conf)))
                    last_fire = now
                    history.clear()
                    last_best = best
        
        except Exception as e:
            print("Audio Loop Error:", e)
        finally:
            stream.stop()
            stream.close()

    def process_queue(self):
        try:
            while True:
                msg_type, data = self.msg_queue.get_nowait()
                
                if msg_type == "status_update":
                    label, conf = data
                    # Update status label (mostly for silence/noise)
                    # Don't overwrite if it was a command immediately? 
                    # Actually, user wants to see "SILENCE" and "NOISE"
                    if label in ["silence", "noise"]:
                        self.lbl_status.config(text=label.upper(), foreground="gray")
                        self.reset_arrows() # Clear arrows if silence/noise
                    elif label in COMMANDS:
                         # Don't reset instantly, wait for trigger, or show generic
                         pass
                    
                    self.lbl_confidence.config(text=f"Conf: {conf:.2f}")

                elif msg_type == "command":
                    cmd, conf = data
                    self.trigger_command(cmd)
                    self.lbl_status.config(text=f"DETECTED: {cmd.upper()}", foreground="green")

                elif msg_type == "error":
                    messagebox.showerror("Error", data)

        except queue.Empty:
            pass
        
        self.root.after(50, self.process_queue)

    def trigger_command(self, cmd):
        # Visual
        self.reset_arrows()
        if cmd in self.arrows:
            self.arrows[cmd].config(bg=self.active_bg)
            
            # Reset color after a short delay
            self.root.after(300, lambda c=cmd: self.arrows[c].config(bg=self.default_bg))

        # Keyboard
        if params_keyboard and keyboard:
            key_to_press = KEY_MAPPING.get(cmd)
            if key_to_press:
                # Simulate press and release
                keyboard.press(key_to_press)
                keyboard.release(key_to_press)
                print(f"Simulated Key: {key_to_press}")
            else:
                # If mapped to a string char
                # Handle later in settings
                pass

    def reset_arrows(self):
        for arrow in self.arrows.values():
            arrow.config(bg=self.default_bg)

    def open_settings(self):
        # Settings window setup
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Key Bindings")
        
        # Dimensions
        w, h = 300, 250
        
        # Calculate center relative to parent
        try:
            # Update pending geometry tasks to get accurate values
            self.root.update_idletasks() 
            x_parent = self.root.winfo_x()
            y_parent = self.root.winfo_y()
            w_parent = self.root.winfo_width()
            h_parent = self.root.winfo_height()
            
            x = x_parent + (w_parent - w) // 2
            y = y_parent + (h_parent - h) // 2
            
            settings_win.geometry(f"{w}x{h}+{x}+{y}")
        except:
            # Fallback if calculation fails
            settings_win.geometry(f"{w}x{h}")
            
        settings_win.resizable(False, False)

        tk.Label(settings_win, text="Click prompt to set key for command", font=('Arial', 10)).pack(pady=10)

        for cmd in COMMANDS:
            f = tk.Frame(settings_win)
            f.pack(pady=5, fill=tk.X, padx=20)
            tk.Label(f, text=cmd.upper(), width=10).pack(side=tk.LEFT)
            
            # Display current key
            current_key = KEY_MAPPING.get(cmd)
            lbl = tk.Label(f, text=str(current_key), fg="blue")
            lbl.pack(side=tk.LEFT, padx=10)
            
            btn = tk.Button(f, text="Change", command=lambda c=cmd, l=lbl, p=settings_win: self.change_key(c, l, p))
            btn.pack(side=tk.RIGHT)

    def change_key(self, cmd, label_widget, parent):
        # Custom dialog for capturing key press
        dlg = tk.Toplevel(parent)
        dlg.title(f"Map {cmd.upper()}")
        w, h = 300, 150
        
        # Center relative to parent
        try:
            parent.update_idletasks()
            # Calculate position relative to screen, using parent center
            x = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
            y = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
            dlg.geometry(f"{w}x{h}+{x}+{y}")
        except:
            dlg.geometry(f"{w}x{h}")

        dlg.transient(parent)
        dlg.grab_set()
        dlg.focus_set()
        
        msg = tk.Label(dlg, text=f"Press any key for {cmd.upper()}...\n(Press ESC to cancel, Backspace to reset)", font=('Arial', 10))
        msg.pack(expand=True)

        def on_key(event):
            # Check for special keys to cancel/reset
            if event.keysym == 'Escape':
                dlg.destroy()
                return
            if event.keysym == 'BackSpace':
                # Reset to default
                if cmd == "up": KEY_MAPPING["up"] = Key.up if params_keyboard else "up"
                elif cmd == "down": KEY_MAPPING["down"] = Key.down if params_keyboard else "down"
                elif cmd == "left": KEY_MAPPING["left"] = Key.left if params_keyboard else "left"
                elif cmd == "right": KEY_MAPPING["right"] = Key.right if params_keyboard else "right"
                
                label_widget.config(text=str(KEY_MAPPING[cmd]))
                dlg.destroy()
                return

            # Capture char
            new_key = None
            if event.char and len(event.char) == 1 and event.char.isprintable():
                new_key = event.char.lower()
            
            # Update if valid
            if new_key:
                KEY_MAPPING[cmd] = new_key
                label_widget.config(text=new_key)
                dlg.destroy()
            else:
                 messagebox.showwarning("Invalid Key", "Please press a printable character (a-z, 0-9).", parent=dlg)

        dlg.bind("<Key>", on_key)
        parent.wait_window(dlg)


def main():
    root = tk.Tk()
    app = VoiceCommandApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
