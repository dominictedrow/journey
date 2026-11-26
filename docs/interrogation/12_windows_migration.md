# Questioning: Phase 12 · Windows migration

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/12_windows_migration.md`, covering interfaces, config keys, tests and acceptance criteria.
> Spec reference: —

1. Windows version, CPU, RAM, disk layout, and where the project should live (e.g. `C:\dev\quant_trader`)?
2. Docker Desktop (WSL2 backend), or Postgres native on Windows?
3. Train natively on Windows or in WSL2? (Recommended: native; WSL2 only if Triton issues appear.)
4. How do data, models and the DB dump move across (external drive, network share, cloud)?
5. Does the Mac keep running anything after the migration, or does Windows take over completely?
6. Should the Windows PC run 24/7 with auto-login, Task Scheduler at boot, and wake timers?
