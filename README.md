Model implementation is based on https://github.com/SeanNaren/deepspeech.pytorch

## Run docker

```shell
docker-compose up --build --detach

docker-compose run --rm deep_lt bash
```

### Generate manifest

```shell
python data/local.py --source-dir ../data/liepa_100/ --manifest-dir ./manifests --manifest-name liepa --split 0.7 0.2 0.1
```

### Start training

```shell
python train.py \
--train-manifest manifests/train_liepa.csv \
--val-manifest manifests/val_liepa.csv \
--batch-size 12 \
--comet-offline \
--fast-dev-run
```