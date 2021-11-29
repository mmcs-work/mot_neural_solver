pip install tensorboard==2.4.0 neptune-tensorboard==0.5.1 --use-feature=2020-resolver
export NEPTUNE_API_TOKEN='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YmExZmZiOC1hNmUyLTRiYWMtYWZhMC05YzdmNTQxZDJjZDIifQ=='
neptune tensorboard ./output/experiments/ --project mainak/adl4cv