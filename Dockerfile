FROM python:3.9-slim as builder
COPY . /pkg
WORKDIR pkg
RUN pip install build && \
    python -m build .

FROM python:3.9-slim as trainer
COPY --from=builder /pkg ./
RUN python -m pip install dist/paa-0.1.0-py3-none-any.whl && \
    paa_train
EXPOSE 8080
CMD paa_deploy --path_to_model model.bin