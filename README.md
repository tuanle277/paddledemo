# PaddleOCR PDF Parsing POC

This repo provides end-to-end demos for extracting text and structure from PDFs using PaddleOCR 3.x:
	•	Basic OCR (text + bounding boxes)
	•	PP-StructureV3 (layout → Markdown/JSON)
	•	PaddleOCR-VL (VLM document parsing → Markdown/JSON)
	•	Utilities to render PDF pages to images (for OCR/parsing)

Repo layout

.
├── pdf_to_images.py
├── demo_ocr_pdf.py
├── demo_ppstructurev3_pdf.py
├── demo_paddleocrvl_pdf.py
└── (optional) demo_translation_pdf.py


⸻

1) Prerequisites

Recommended OS / hardware
	•	macOS (Apple Silicon or Intel), Linux (Ubuntu / Amazon Linux) supported
	•	For big PDFs or VLM parsing: more CPU/RAM helps

Python
	•	Recommended: Python 3.10 (best compatibility for PaddleOCR + PaddlePaddle)
	•	Works often on 3.11, but 3.10 is safest

Conda (recommended)

Miniconda / Anaconda / Mambaforge is fine.

⸻

2) Environment setup (macOS + Conda)

2.1 Create and activate env

conda create -n paddle-ocr python=3.10 -y
conda activate paddle-ocr
python -V

2.2 Install dependencies (pip)

Upgrade pip tooling first:

python -m pip install -U pip setuptools wheel

Install core packages:

Option A (recommended): install PaddleOCR with “doc parser” features
This installs PaddleOCR + its doc parsing dependencies.

python -m pip install "paddleocr[doc-parser]"

Option B: install everything
If you want all features (doc parsing, translation, IE, etc.):

python -m pip install "paddleocr[all]"

Add the rest (PDF rendering + visualization)

python -m pip install pypdfium2 numpy pillow

Notes:
	•	pypdfium2 is used to render PDF pages to images reliably.
	•	You can add matplotlib if you want to visualize results locally.

2.3 (Optional) If you also use LangChain/OpenAI in the same env

python -m pip install langchain==0.1.20 langchain-core==0.1.52 langchain-openai==0.1.7 openai


⸻

3) Environment setup (AWS EC2)

3.1 Instance recommendations

For CPU-only OCR/parsing:
	•	t3.medium (works for simple OCR, can be slow)
	•	t3.large / m6i.large (better)
	•	If using VLM / heavy parsing, bigger is better.

3.2 Install system deps (Ubuntu 24.04)

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git

(Optional but useful for stability/performance)

sudo apt-get install -y libgl1 libglib2.0-0

3.3 Create venv (recommended on EC2)

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

3.4 Install PaddleOCR + PDF deps

python -m pip install "paddleocr[doc-parser]" pypdfium2 numpy pillow

If you want everything:

python -m pip install "paddleocr[all]" pypdfium2 numpy pillow


⸻

4) Important runtime environment variables (highly recommended)

PaddleOCR / PaddleX downloads models on first run. These environment variables help avoid timeouts and reduce crashes.

4.1 Model download / source selection

export DISABLE_MODEL_SOURCE_CHECK=True
export PADDLE_PDX_MODEL_SOURCE=HF

4.2 Reduce thread contention (prevents some macOS segfault/bus errors)

export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

You can put these into your shell startup or run them inline before commands.

⸻

5) Run commands (PDF → text / Markdown / JSON)

5.1 Convert PDF pages to images (helper)

pdf_to_images.py is imported by the demos; you don’t normally run it directly.

⸻

5.2 Basic OCR on a PDF (text + boxes)

Outputs ocr_results.json with per-page extracted lines.

python demo_ocr_pdf.py --pdf your.pdf --out output_ocr --lang en --dpi 200

Common options:
	•	--dpi 180 (faster) to --dpi 220 (better for small text)
	•	--start 0 --end 5 to process only pages 0..4

Example:

python demo_ocr_pdf.py --pdf your.pdf --out output_ocr --lang en --dpi 200 --start 0 --end 3


⸻

5.3 PP-StructureV3 (layout parsing → Markdown + JSON)

This is typically the best “PDF → structured Markdown” approach.

python demo_ppstructurev3_pdf.py --pdf your.pdf --out out_structure --dpi 200

Outputs:

out_structure/
  page_0000/
    *.json
    *.md
  page_0001/
    *.json
    *.md
  ...


⸻

5.4 PaddleOCR-VL (VLM-based doc parsing → Markdown + JSON)

Heavier but strong on complex docs.

python demo_paddleocrvl_pdf.py --pdf your.pdf --out out_vl --dpi 200


⸻

5.5 (Optional) Translation pipeline

If your PaddleOCR build includes translation pipelines, install:

python -m pip install "paddleocr[trans]"
# or "paddleocr[all]"

Then run:

python demo_translation_pdf.py --pdf your.pdf --out out_translate --src_lang en --tgt_lang zh


⸻

6) Verifying installation

6.1 Sanity checks

python -c "import paddle; import paddleocr; print('paddle', paddle.__version__, 'paddleocr', paddleocr.__version__)"
python -c "import pypdfium2; print('pypdfium2 ok')"

6.2 Confirm doc-parser extras are installed (common failure)

If you see errors like:

DependencyError: OCR requires additional dependencies... pip install "paddlex[ocr]==..."

Run:

python -m pip install -U "paddlex[ocr]" "paddlex[ocr-core]"

In most cases, installing paddleocr[doc-parser] or paddleocr[all] is the cleanest fix:

python -m pip install -U "paddleocr[doc-parser]"
# or
python -m pip install -U "paddleocr[all]"


⸻

7) Troubleshooting (your common errors)

7.1 “FileNotFoundError … inference.yml” in ~/.paddlex/official_models/...

This usually means the model download was interrupted, leaving a partial folder.

Fix:

rm -rf ~/.paddlex/official_models/en_PP-OCRv5_mobile_rec

Then re-run with:

export PADDLE_PDX_MODEL_SOURCE=HF
export DISABLE_MODEL_SOURCE_CHECK=True
python demo_ocr_pdf.py --pdf your.pdf --out output_ocr


⸻

7.2 “segmentation fault” / “bus error” on macOS

This is commonly caused by:
	•	thread over-subscription (OpenMP / Accelerate)
	•	conflicting OpenCV installs (conda vs pip)
	•	mixed binary wheels in one env

Do these in order:

Step 1: limit threads

export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

Step 2: ensure you only have ONE OpenCV
If you installed OpenCV via conda, don’t pip-install another OpenCV.

Check:

python -c "import cv2; print(cv2.__version__, cv2.__file__)"
conda list | grep -i opencv
python -m pip show opencv-python opencv-contrib-python opencv-python-headless

If conda owns OpenCV, remove via conda:

conda remove -y opencv

Then (optional) install via pip (one of them only):

python -m pip install opencv-python-headless

Step 3: new clean environment
If it still crashes, fastest fix is a clean env:

conda create -n paddle-ocr-clean python=3.10 -y
conda activate paddle-ocr-clean
python -m pip install -U pip setuptools wheel
python -m pip install "paddleocr[doc-parser]" pypdfium2 numpy pillow


⸻

7.3 “Checking connectivity to the model hosters…” hangs or times out

Use:

export DISABLE_MODEL_SOURCE_CHECK=True
export PADDLE_PDX_MODEL_SOURCE=HF

If your network blocks one source, this forces HuggingFace.

⸻

7.4 pip uninstall opencv-python ... no RECORD file

That means OpenCV was installed by conda, not pip.

Uninstall with:

conda remove -y opencv opencv-python


⸻

7.5 “wheel requires packaging>=24.0 but you have packaging 23.2”

If you pinned langchain-core==0.1.52 it pulls packaging<24. Fix by letting pip resolve, or pin wheel lower, but the simplest is:

python -m pip install -U "pip<26" setuptools wheel packaging

Or keep a dedicated OCR-only environment without LangChain.

⸻

8) Recommended “two environment” workflow (best practice)

To avoid dependency fights:

Env 1: OCR only (stable)
	•	paddleocr[doc-parser]
	•	pypdfium2 numpy pillow

Env 2: LLM / LangChain
	•	langchain* openai
	•	your other tooling

This prevents packaging / pydantic / OpenCV conflicts.

⸻

9) Example: end-to-end run (recommended)

conda activate paddle-ocr

export DISABLE_MODEL_SOURCE_CHECK=True
export PADDLE_PDX_MODEL_SOURCE=HF
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python demo_ppstructurev3_pdf.py --pdf your.pdf --out out_structure --dpi 200 --start 0 --end 3


⸻

10) What to commit / what not to commit

Do not commit:
	•	~/.paddlex/official_models/... (model cache)
	•	huge outputs

Suggested .gitignore:

output_ocr/
out_structure/
out_vl/
output_translation/
*.pdf
# paddledemo
