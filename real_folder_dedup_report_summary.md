# Real Folder Dedup Report (Run Summary)

- Source: `D:\NOW SORT\Duplicates Backup`
- Move target: `D:\NOW SORT\Extera`
- Run folder: `D:\NOW SORT\Extera\dedup_run_20260320_035720`
- Audio files scanned: **264**
- Exact duplicate groups: **0**
- Strict audio match pairs: **2373**
- Near-match review candidates: **0**
- Files moved: **257**
- Remaining audio files in source: **7**
- Runtime: **116.75s**

## Output files
- JSON report: `real_folder_dedup_report.json`
- CSV report: `real_folder_dedup_report.csv`

## Rollback (if needed)
Use:

```powershell
.\.venv-app\Scripts\python.exe restore_dedup_moves.py --report real_folder_dedup_report.json
```

Dry run:

```powershell
.\.venv-app\Scripts\python.exe restore_dedup_moves.py --report real_folder_dedup_report.json --dry-run
```
