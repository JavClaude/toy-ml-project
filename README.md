# paa 

A toy package to train and deploy a baseline iris classifier


To install the package, please run the following command
```bash
pip install build # dependency needed to build our package
python -m build . # build the wheel
pip install dist/paa-0.1.0-py3-none-any.whl # install the wheel
```

## Train an Iris classifier

Once the package is installed you can run a training with ``paa_train`` entrypoint:
```bash
paa_train --path_to_save_model <your path here>
          --test_size
          --n_estimators
          --max_depth
          --random_state_splitter
          --random_state_classifier
```

## Serve the trained model

Use the following entrypoint to server your model though an api
```bash
paa_deploy --path_to_model
           --host
           --port
```

## Code quality and tests

```bash
flake8 paa
black paa
pytest 
```
