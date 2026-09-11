An older classifier can leave a complete provider rejection recorded as an unknown paid outcome. After deploying the corrected engine, reconcile that original run with:

```sh
uv run python -m backend.scripts.reconcile_paid_run --data-dir /absolute/path/to/engine/data ORIGINAL_RUN_ID
```

Use the existing data directory and original run ID. The command takes the normal owner lock and verifies the saved receipt hashes and admission bindings. It makes no provider requests. Complete rejections become rejected receipts; successful responses stay saved. Missing, changed, or ambiguous evidence leaves work blocked and exits nonzero.

On success, the existing runtime resumes the original confirmed intent, including any remaining paid calls. Previously saved requests are replayed from their receipts. The normal partial-council policy applies: a rejected member remains failed while available members continue. An interrupted reconciliation can be resumed with the same command.
