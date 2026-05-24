import torch, cv2, numpy as np, tkinter as tk
from tkinter import filedialog
import threading
from PIL import Image, ImageTk
from model import ViolenceDetector

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '../models/best_model.pth'
NUM_FRAMES, IMG_SIZE = 16, 224

def load_model():
    m = ViolenceDetector().to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.eval()
    return m

def preprocess_frame(frame):
    f = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    f = f/255.0
    return np.transpose(f,(2,0,1))

# ─── Upload Video UI ───────────────────────────────
class UploadUI(tk.Toplevel):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.title("Upload Video Detection")
        self.geometry("1100x750")
        self.configure(bg="#0f172a")

        tk.Label(self,text="📂 Upload Video Detection",
                 font=("Segoe UI",22,"bold"),
                 bg="#0f172a",fg="#38bdf8").pack(pady=15)

        self.canvas = tk.Canvas(self,bg="#1e293b",width=960,height=540,highlightthickness=0)
        self.canvas.pack(expand=True)

        # Confidence bar
        self.conf_bar = tk.Canvas(self,bg="#1e293b",height=30,width=960,highlightthickness=0)
        self.conf_bar.pack(pady=10)

        self.status = tk.Label(self,text="Awaiting prediction...",
                               font=("Segoe UI",16,"bold"),
                               bg="#1e293b",fg="#f1f5f9",height=2)
        self.status.pack(fill="x",pady=15)

        tk.Button(self,text="Select Video",command=self._select_video,
                  font=("Segoe UI",14,"bold"),
                  bg="#38bdf8",fg="#0f172a",width=25).pack(pady=15)

    def _select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video","*.mp4 *.avi *.mov *.mkv")])
        if path: threading.Thread(target=self._play_video,args=(path,),daemon=True).start()

    def _play_video(self,path):
        cap=cv2.VideoCapture(path); buf=[]; probs=[]
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000/fps) if fps>0 else 33

        def loop():
            ret,frame=cap.read()
            if not ret:
                cap.release()
                return
            buf.append(preprocess_frame(frame))
            if len(buf)>NUM_FRAMES: buf.pop(0)
            if len(buf)==NUM_FRAMES:
                arr=np.array(buf,dtype=np.float32)
                t=torch.tensor(arr).unsqueeze(0).to(DEVICE)
                with torch.no_grad(): prob=self.model(t).item()
                probs.append(prob)
                avg_prob = np.mean(probs[-10:])
                label="VIOLENT" if avg_prob>=0.5 else "NON-VIOLENT"
                self._update_display(frame,label,avg_prob*100)
            self.after(delay, loop)

        loop()

    def _update_display(self,frame,label,pct):
        h,w,_=frame.shape; scale=min(960/w,540/h)
        frame=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
                         (int(w*scale),int(h*scale)))
        x1,y1,x2,y2=int(10*scale),int(10*scale),int((w-10)*scale),int((h-10)*scale)
        cv2.rectangle(frame,(x1,y1),(x2,y2),
                      (0,255,0) if label=="NON-VIOLENT" else (255,0,0),3)
        cv2.putText(frame,f"{label} ({pct:.1f}%)",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,
                    (0,255,0) if label=="NON-VIOLENT" else (255,0,0),2)
        img=ImageTk.PhotoImage(Image.fromarray(frame))
        self.canvas.create_image(480,270,image=img)
        self.canvas.image=img
        self.status.config(text=label)

        # Confidence bar
        self.conf_bar.delete("all")
        bar_len = int((pct/100)*960)
        color = "#22c55e" if label=="NON-VIOLENT" else "#ef4444"
        self.conf_bar.create_rectangle(0,0,bar_len,30,fill=color)

# ─── Real-time Webcam UI ───────────────────────────
class RealtimeUI(tk.Toplevel):
    def __init__(self, model):
        super().__init__()
        self.model=model
        self.title("Real-time Webcam Detection")
        self.geometry("1100x750")
        self.configure(bg="#0f172a")

        tk.Label(self,text="🎥 Real-time Webcam Detection",
                 font=("Segoe UI",22,"bold"),
                 bg="#0f172a",fg="#4ade80").pack(pady=15)

        self.canvas=tk.Canvas(self,bg="#1e293b",width=960,height=540,highlightthickness=0)
        self.canvas.pack(expand=True)

        self.conf_bar = tk.Canvas(self,bg="#1e293b",height=30,width=960,highlightthickness=0)
        self.conf_bar.pack(pady=10)

        self.status=tk.Label(self,text="Awaiting prediction...",
                             font=("Segoe UI",16,"bold"),
                             bg="#1e293b",fg="#f1f5f9",height=2)
        self.status.pack(fill="x",pady=15)

        tk.Button(self,text="Start Webcam",command=self._start,
                  font=("Segoe UI",14,"bold"),
                  bg="#4ade80",fg="#0f172a",width=25).pack(pady=10)

        tk.Button(self,text="Stop Webcam",command=self._stop,
                  font=("Segoe UI",14,"bold"),
                  bg="#f87171",fg="#0f172a",width=25).pack(pady=5)

        self.running=False

    def _start(self):
        self.running=True
        threading.Thread(target=self._run,daemon=True).start()

    def _stop(self):
        self.running=False

    def _run(self):
        cap=cv2.VideoCapture(0); buf=[]; probs=[]
        while self.running:
            ret,frame=cap.read()
            if not ret: break
            frame=cv2.flip(frame,1)
            buf.append(preprocess_frame(frame))
            if len(buf)>NUM_FRAMES: buf.pop(0)
            if len(buf)==NUM_FRAMES:
                arr=np.array(buf,dtype=np.float32)
                t=torch.tensor(arr).unsqueeze(0).to(DEVICE)
                with torch.no_grad(): prob=self.model(t).item()
                probs.append(prob)
                avg_prob = np.mean(probs[-10:])
                label="VIOLENT" if avg_prob>=0.5 else "NON-VIOLENT"
                self._update_display(frame,label,avg_prob*100)
            cv2.waitKey(30)  # slow down webcam loop
        cap.release()

    def _update_display(self,frame,label,pct):
        h,w,_=frame.shape; scale=min(960/w,540/h)
        frame=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
                         (int(w*scale),int(h*scale)))
        x1,y1,x2,y2=int(10*scale),int(10*scale),int((w-10)*scale),int((h-10)*scale)
        cv2.rectangle(frame,(x1,y1),(x2,y2),
                      (0,255,0) if label=="NON-VIOLENT" else (255,0,0),3)
        cv2.putText(frame,f"{label} ({pct:.1f}%)",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,
                    (0,255,0) if label=="NON-VIOLENT" else (255,0,0),2)
        img=ImageTk.PhotoImage(Image.fromarray(frame))
        self.canvas.create_image(480,270,image=img)
        self.canvas.image=img
        self.status.config(text=label)

                # Confidence bar
        self.conf_bar.delete("all")
        bar_len = int((pct/100)*960)
        color = "#22c55e" if label=="NON-VIOLENT" else "#ef4444"
        self.conf_bar.create_rectangle(0,0,bar_len,30,fill=color)

# ─── Launcher ──────────────────────────────────────
class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Violence Detection Launcher")
        self.geometry("450x300")
        self.configure(bg="#0f172a")

        tk.Label(self,text="Violence Detection System",
                 font=("Segoe UI",20,"bold"),
                 bg="#0f172a",fg="#facc15").pack(pady=20)

        model=load_model()

        tk.Button(self,text="📂 Upload Video Detection",
                  font=("Segoe UI",14,"bold"),
                  bg="#38bdf8",fg="#0f172a",
                  command=lambda:UploadUI(model)).pack(pady=10)

        tk.Button(self,text="🎥 Real-time Webcam Detection",
                  font=("Segoe UI",14,"bold"),
                  bg="#4ade80",fg="#0f172a",
                  command=lambda:RealtimeUI(model)).pack(pady=10)

        tk.Button(self,text="❌ Exit",
                  font=("Segoe UI",14),
                  bg="#f87171",fg="#0f172a",
                  command=self.destroy).pack(pady=10)

# ─── Entry Point ──────────────────────────────────────
if __name__=="__main__":
    app = Launcher()
    app.mainloop()
