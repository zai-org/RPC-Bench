<div align="center">

# RPC-Bench: A Fine-grained Benchmark for Research Paper Comprehension

</div>

<p align="center">
    🌐 <a href="https://rpc-bench.github.io/" target="_blank">Project Page</a> •
    📖 <a href="https://arxiv.org/abs/2601.14289" target="_blank">Paper</a>
    🤗 <a href="https://huggingface.co/datasets/zai-org/RPC-Bench" target="_blank">Hugging Face</a> •
    🧭 <a href="https://modelscope.cn/datasets/ZhipuAI/RPC-Bench" target="_blank">ModelScope</a>
</p>

<div align="center">
    <img src=assets/pipeline.png width=100% />
</div>

*Official code and data of the paper RPC-Bench: A Fine-grained Benchmark for Research Paper Comprehension (ACL 2026).*

***

RPC-Bench, a large-scale fine-grained question answering benchmark constructed from review-rebuttal exchanges of high-quality academic papers, with each paper available in two input formats (pure text and rendered page images) enabling evaluation of both large language models (LLMs) and visual language models (VLMs).

## 🚀 Quick Start

### Dependencies

First, create a conda environment and install all pip package requirements.

```bash
conda create -n rpc python==3.11.13
conda activate rpc

pip install -r requirements.txt
```

### QA Construction

The [`pipeline/`](pipeline/) directory provides an example workflow for constructing benchmark QA annotations from crawled OpenReview review-rebuttal data through LLM-based decomposition, rewriting, and filtering.
See [`pipeline/README.md`](pipeline/README.md) for details.

### Data processing

For this benchmark, each academic paper can be processed into either **structured text** or **page-rendered images**, enabling evaluation across both LLMs and VLMs. Choose the parsing mode that best fits your experimental objectives.

- **File Download**: Download paper PDFs based on metadata from JSON files located under the `benchmark/` directory.
```bash
python download.py
```
- **Text Parsing**: Parse PDF content into text using [MinerU](https://github.com/opendatalab/MinerU).
```bash
pip install --upgrade pip
pip install uv
uv pip install -U "mineru[core]"

mineru-models-download
mineru -p "./benchmark/pdf/test" -o "./benchmark/parse/test" --source local
```
- **Image Parsing**: Convert PDF pages into image format for further processing.
```bash
python pdf2image.py
```

#### Processed Data Download

You may also download our processed data directly from [Google Drive](https://drive.google.com/drive/folders/1J__Hp0PPE6VATlyMRaM6dIZST8H1zTPj?usp=sharing), [Hugging Face](https://huggingface.co/), or [ModelScope](https://www.modelscope.cn/).
The processed data includes:

- `pdf/`: original paper PDFs.
- `md/`: Markdown files parsed from each paper by MinerU, used as text input for LLM-oriented evaluation.
- `parse/`: full MinerU parsing outputs, including structured layout and content artifacts.
- `vlm/`: page images rendered from PDFs with PyMuPDF at 200 DPI, used for VLM-oriented evaluation.

### 🧩 Consistency Evaluation

The [`consistency/`](consistency/) directory provides a self-contained example for measuring consistency between LLM judge outputs and human pairwise preferences.
See [`consistency/README.md`](consistency/README.md) for details.

### ✈️ Inference

GPT-5 is given as an example below, but you may replace this with any other LLM or VLM supported in your environment.

- **LLM Inference**:
```bash
python llm.py
```

- **VLM Inference**:
```bash
python vlm.py
```

### 🛜 Evaluation

After inference, evaluate predictions against benchmark references using:

```bash
python eval.py
```
