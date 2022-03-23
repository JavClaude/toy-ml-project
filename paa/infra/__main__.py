import argparse

import uvicorn

from paa.infra.application.app import create_app


def deploy_model() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--host", type=str, required=False, default="0.0.0.0")
    argument_parser.add_argument("--port", type=int, required=False, default=8080)
    argument_parser.add_argument("--path_to_model", type=str, required=True)

    arguments = argument_parser.parse_args()

    app = create_app(arguments.path_to_model)

    uvicorn.run(app, host=arguments.host, port=arguments.port)
