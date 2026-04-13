ChangeVLM

```
pip uninstall -y mmcv mmcv-full mmcv-lite openmim

python -m pip install -U pip wheel ninja
python -m pip install "setuptools<82" "packaging<25"

export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MAX_JOBS=8

python -m pip install --no-build-isolation -v "mmcv==2.2.0"
```
