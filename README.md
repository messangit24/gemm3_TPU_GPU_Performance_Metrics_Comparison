# LLM Inference Dashboard (Gemma 3 27B)

A Streamlit-based chat interface and performance telemetry dashboard for monitoring Large Language Models deployed on Google Kubernetes Engine (GKE) using Google Cloud Accelerators (TPUs and GPUs).

## Architecture Overview
This project routes user prompts to various `vLLM` endpoints hosted on GKE and calculates real-time performance metrics for each inference request. 

**Supported Hardware Endpoints:**
* 4x Google Cloud TPU v6e (Trillium)
* 4x NVIDIA A100 / H100 GPUs
* 8x Google Cloud TPU v6e (Trillium)

## Features
* **Live Inference Chat:** Interacts directly with vLLM APIs using the OpenAI-compatible Python client.
* **Document Context:** Supports uploading `.pdf`, `.txt`, and `.csv` files to inject context directly into the LLM prompt.
* **Real-time Telemetry:** Calculates TTFT (Time to First Token), TPOT (Time Per Output Token), and Throughput (Tokens/Sec).
* **Hardware Utilization:** Mathematically derives Model FLOPs Utilization (MFU) and Memory Bandwidth Utilization (MBU), while pulling live physical duty-cycle data (MXU / SM Utilization) directly from Google Cloud Monitoring.
* **BigQuery Logging:** Every query and its associated hardware telemetry is securely logged to a BigQuery dataset (`tpu_metrics`) for historical trend analysis.

---

## Hardware & Infrastructure Specifications

To run Gemma 3 27B across different accelerator architectures in GKE, specific container images, resource limits, and node selectors are required for the `vLLM` engine.

### 1. TPU v6e Deployment (4x Chips)
* **Container Image:** `vllm/vllm-tpu:latest`
* **Model Name:** `google/gemma-3-27b-it`
* **Node Selector:** `cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice` (or equivalent topology label based on your GKE setup).
* **Resource Limits:** `google.com/tpu: "4"`
* **vLLM Arguments:** * `--tensor-parallel-size=4`
  * `--max-model-len=8192`
  * *(A Hugging Face token is required as an environment variable `HF_TOKEN` to pull the weights).*

### 2. NVIDIA A100 / H100 Deployment (4x GPUs)
* **Container Image:** `vllm/vllm-openai:latest`
* **Model Name:** `google/gemma-3-27b-it`
* **Node Selector:** `cloud.google.com/gke-accelerator: nvidia-tesla-a100` (or `nvidia-h100-80gb`)
* **Resource Limits:** `nvidia.com/gpu: "4"`
* **Volume Mounts:** Requires an `emptyDir` mount at `/dev/shm` (`medium: Memory`) to provide sufficient shared memory for GPU tensor operations.
* **vLLM Arguments:**
  * `--tensor-parallel-size=4`
  * `--gpu-memory-utilization=0.9`
  * `--max-model-len=8192`

---

## Setup & Deployment

**1. Environment Variables**
The dashboard expects the following environment variables to securely connect to GCP and your internal GKE services:
* `PROJECT_ID`: Your GCP Project ID.
* `CLUSTER_NAME`: The name of the GKE cluster hosting the vLLM pods.
* `BQ_DATASET`: (Default: `tpu_metrics`) The BigQuery dataset for logs.

**2. Local Execution**
```bash
pip install -r requirements.txt
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
