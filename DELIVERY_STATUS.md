# Delivery status

- Student: Muhammad Anas Murtaza, DF2-AI-2026-031.
- Real CIFAR-100 download checksum verified.
- CNN trained from scratch for 30 epochs on the assigned 5-class subset.
- Selected checkpoint: epoch 29 (validation accuracy).
- Official test result: 70.60% (353/500 correct).
- Split: 2000 training, 500 validation, 500 test.
- models/insect_cnn.pt contains actual trained weights, not a demo or random checkpoint.
- Report contains actual curves, confusion matrix, three measured mistakes and real Tkinter screenshots.
- GUI tested on a virtual Linux desktop: prediction, top-3, save, history preview, logout.
- Database/model unit tests passed. Windows launcher supplied, but Windows itself was not available for testing.
- Training runtime: torch 2.15.0+cpu, numpy 2.3.5.
- Dataset, personal accounts, local database, copied user images, virtual environments and caches are excluded from the ZIP.
- Sample GUI screenshots use a temporary demonstration account. No credentials are shipped.

Read PROJECT_EXPLAINED_ROMAN_URDU.md and be able to explain all submitted code.
