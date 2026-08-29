# FiberVision update bundle

Expected GitHub main:
`f4e7becc0a4193e14b18a9e96d12428c935994b2`

On the EC2 server:

```bash
unzip -o ~/FiberVision_auth_review_results_export_update.zip -d ~/FiberVision_auth_review_results_export_update
bash ~/FiberVision_auth_review_results_export_update/apply_and_push.sh
```

The script refuses to run if the repository is dirty or HEAD no longer matches the expected baseline.
After pushing, GitHub CI must pass before the existing deployment workflow updates EC2.

After deployment, create the first account:

```bash
cd ~/FiberVision
docker compose exec api python -m app.cli.create_user YOUR_EMAIL
```

The CLI asks for the initial password twice. The user must change it on first login.
