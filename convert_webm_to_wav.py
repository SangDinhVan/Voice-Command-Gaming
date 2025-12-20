import subprocess
from pathlib import Path
import shutil

# ====== CONFIG ======
SRC_DIR = Path(r"S:\Downloads\noise")              # nơi chứa webm
DST_DIR = Path(r"data\raw\gg2")     # nơi noise chuẩn cho project
SAMPLE_RATE = 16000
# ====================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def convert_one(src: Path, dst: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),
        "-ac", "1",                 # mono
        "-ar", str(SAMPLE_RATE),    # 16kHz
        "-c:a", "pcm_s16le",        # WAV PCM 16-bit
        str(dst)
    ]
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

def main():
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Không thấy thư mục {SRC_DIR}")

    ensure_dir(DST_DIR)

    files = list(SRC_DIR.glob("*.webm"))
    if not files:
        print("❌ Không có file .webm nào trong thư mục noise")
        return

    print(f"🔍 Found {len(files)} webm files")

    for i, f in enumerate(files):
        out = DST_DIR / f"{f.stem}.wav"
        try:
            convert_one(f, out)
            print(f"[{i+1}/{len(files)}] OK: {f.name} -> {out.name}")
        except Exception as e:
            print(f"[FAIL] {f.name}: {e}")

    print("\n✅ DONE: Noise đã được convert và copy vào project")
    print(f"📁 {DST_DIR}")

if __name__ == "__main__":
    main()
