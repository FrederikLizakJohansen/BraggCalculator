# COD throughput benchmark commands

Run these commands from the repository root on the A3000 machine.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/benchmark_cif_corpus.py \
  --backend numpy --device cpu \
  --output paper/data/cod_throughput_cpu_numpy_A3000.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/benchmark_cif_corpus.py \
  --backend torch --device cpu \
  --output paper/data/cod_throughput_cpu_torch_A3000.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/benchmark_cif_corpus.py \
  --backend torch --device cuda \
  --output paper/data/cod_throughput_gpu_torch_A3000.json
```
