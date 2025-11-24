# Metalizer
Convert ANY song into Heavy Metal

## Backend Scaffold Overview

```
Metalizer/
├── app/
│   ├── api/
│   │   └── routes.py           # FastAPI routers and request models
│   ├── core/
│   │   └── config.py           # Settings, paths, environment toggles
│   ├── services/
│   │   └── processor.py        # MetalizerPipeline orchestration logic
│   ├── utils/
│   │   └── audio_io.py         # Shared helpers (file validation, temp dirs)
│   └── main.py                 # FastAPI app entrypoint
├── assets/
│   ├── drums/                  # High-quality double-kick drum samples
│   └── midi/                   # MIDI motifs, riffs, tempo templates
├── data/
│   ├── inputs/                 # Uploaded source tracks (per-session)
│   ├── outputs/                # Rendered heavy metal remixes
│   └── samples/                # Reference stems for experiments/tests
├── models/                     # Pre-trained separation/backbone checkpoints
├── requirements.txt            # Python dependencies
└── README.md
```

This layout keeps runtime artifacts (`data/*`) isolated from code, makes FastAPI modules discoverable, and creates clear homes for DSP assets and ML checkpoints.

## Job Processing & Assets

- **Persistent jobs**: Every remix request is tracked in `data/jobs.sqlite3`. Metadata survives restarts and old rows are pruned after 24 hours along with their rendered stems.
- **Queue backends**: By default jobs run locally, but setting `QUEUE_BACKEND=rq` and `REDIS_URL=redis://...` routes work to an RQ queue (see `app/worker.py`).
- **Job API**: `/api/jobs`, `/api/jobs/{id}`, and `/api/jobs/{id}/result` expose status, download links, and deletion.
- **Asset management**: Use `/api/assets` and `/api/assets/upload` to inspect or extend the drum/guitar sample banks without redeploying.

## Deploying to Fly.io

1. **Install the Fly CLI**
	```bash
	curl -L https://fly.io/install.sh | sh
	fly auth login
	```

2. **Seed the Fly config**
	```bash
	cp fly.example.toml fly.toml
	# edit fly.toml -> set `app`, `primary_region`, and any env overrides
	```

3. **Provision persistent storage** (used for uploads, renders, and SQLite fallbacks)
	```bash
	fly volumes create metalizer_data --size 10 --region <your-region>
	```

4. **Configure secrets / env vars**
	```bash
	fly secrets set \
	  METALIZER_ASSETS_API_KEY=<random> \
		 METALIZER_REDIS_URL=redis://default:pass@host:port/0 \
	  METALIZER_JOBS_DB_URL=postgresql+psycopg://user:pass@host:5432/metalizer
	```
	- Attach your preferred Redis and Postgres services (Fly Postgres, Upstash, etc.).
	 - The `fly.example.toml` maps `METALIZER_REDIS_URL` → `REDIS_URL` so both the API (RQ enqueue) and the worker (`rq worker --url ...`) hit the same instance; don’t leave it unset or the worker will loop/crash.
	 - Assets directory defaults to `/app/assets`; mount additional packs via volumes or object storage if needed.

5. **Deploy the API**
	```bash
	fly deploy
	```

6. **Scale workers (RQ)**
	```bash
	fly scale count app=1
	fly scale count worker=1
	```
	The sample `fly.example.toml` defines `app` (uvicorn) and `worker` (RQ) process groups. Adjust counts or add machines per workload.

### Notes

- The Docker image installs `ffmpeg` and binds `/data` for uploads/outputs. Always attach the `metalizer_data` volume (or equivalent) so rendered mixes persist.
- If you prefer SQLite for jobs, keep the default `METALIZER_JOBS_DB_URL=sqlite:////data/jobs.sqlite3`. Otherwise switch to Postgres for multi-instance deployments.
- Use `fly secrets set METALIZER_ASSET_MAX_BYTES=52428800` (for example) to enforce tighter upload limits per environment.
