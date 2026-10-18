# Public hosting — Clearity

The submission deployment uses a Render Python Web Service with a CPU-only ResNet18 temporal binary classifier. See the English [README](README.md) and [submission checklist](SUBMISSION.md) for the API, startup configuration, measured results and remaining gaps.

## Hosted service

The deployment runs independently of the developer's laptop. Qwen is disabled on the free Render instance because of its memory requirements. All video inference takes place inside the deployed service; no external AI API receives frames.

Use the verified HTTPS base URL listed in SUBMISSION.md. Enter the privately supplied jury key in the web interface, or send it as X-API-Key. Never put the key in a URL or commit it to GitHub.

Free Render instances sleep after inactivity, so a cold request can take 50 seconds or more to wake the server. Jobs and evidence are stored on ephemeral disk and can be lost on redeploy. A technical failure is returned as failed rather than a negative verdict.

The public application hides the local dataset and annotation routes. Training videos, original labels and the local annotation database are not part of the deployment.

## Optional laptop tunnel

The older start-public.ps1 and start-tunnel.ps1 workflow remains available for local demonstrations. A tunnel stops when the laptop or its server stops; it is not the independent hosted submission. The optional start-model.ps1 process enables local Qwen observations through /api/v1/inspect.
