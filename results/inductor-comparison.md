# mlc vs Inductor: what each one generates

device `cuda`, dtype `float16`, seq 128

| model | batch | compiler | generated kernels | pointwise | reduction | of which persistent | extern calls |
|---|---:|---|---:|---:|---:|---:|---:|
| bert-base | 1 | mlc | 100 | 63 | 37 | 37 | 100 |
| bert-base | 1 | inductor | 6 | 4 | 2 | 2 | 73 |
| bert-base | 32 | mlc | 136 | 99 | 37 | 37 | 100 |
| bert-base | 32 | inductor | 6 | 4 | 2 | 2 | 73 |
| gpt2-small | 1 | mlc | 73 | 36 | 37 | 37 | 75 |
| gpt2-small | 1 | inductor | 8 | 4 | 4 | 3 | 49 |
| gpt2-small | 32 | mlc | 109 | 72 | 37 | 37 | 75 |
| gpt2-small | 32 | inductor | 6 | 2 | 4 | 3 | 49 |
