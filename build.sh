pip install build
python -m build .
pip install dist/paa-0.1.0-py3-none-any.whl
paa_train
paa_deploy --path_to_model model.bin
