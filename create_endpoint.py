import asyncio
from webbrowser import open as browse
import argparse
from trainml.trainml import TrainML


parser = argparse.ArgumentParser(description="Stable Diffusion 2 Endpoint")

parser.add_argument(
    "model_id",
    type=str,
    help="The ID of the trainML Model containing the stable diffusion code and weights",
)

parser.add_argument(
    "--model-type",
    type=str,
    choices=["depth", "upscaling", "inpainting"],
    default="upscaling",
    help="The image transformation model to launch",
)

parser.add_argument(
    "--server-type",
    type=str,
    choices=["gradio", "streamlit"],
    default="streamlit",
    help="The endpoint server implementation to launch",
)


async def launch_endpoint(trainml_client, model_id, server_type, model_type):
    env = [
        dict(key="HUGGINGFACE_HUB_CACHE", value="/opt/trainml/models"),
    ]
    if server_type == "gradio":
        env.append(dict(key="GRADIO_SERVER_PORT", value="80"))
        env.append(dict(key="GRADIO_SERVER_NAME", value="0.0.0.0"))
    if server_type == "streamlit":
        env.append(dict(key="STREAMLIT_SERVER_PORT", value="80"))
        env.append(dict(key="STREAMLIT_SERVER_HEADLESS", value="true"))

    models = {
        "depth": {
            "script": "depth2img.py",
            "config": "v2-midas-inference.yaml",
            "checkpoint": "512-depth-ema.ckpt",
        },
        "upscaling": {
            "script": "superresolution.py",
            "config": "x4-upscaling.yaml",
            "checkpoint": "x4-upscaler-ema.ckpt",
        },
        "inpainting": {
            "script": "inpainting.py",
            "config": "v2-inpainting-inference.yaml",
            "checkpoint": "512-inpainting-ema.ckpt",
        },
    }

    start_command = "python " if server_type == "gradio" else "streamlit run "
    start_command += f"$TRAINML_MODEL_PATH/scripts/{server_type}/{models[model_type]['script']} "
    start_command += f"$TRAINML_MODEL_PATH/configs/stable-diffusion/{models[model_type]['config']} "
    start_command += (
        f"$TRAINML_MODEL_PATH/checkpoints/{models[model_type]['checkpoint']} "
    )
    print("starting server with command:", start_command)
    job = await trainml_client.jobs.create(
        f"Stable Diffusion 2 - {model_type} on {server_type}",
        type="endpoint",
        gpu_types=["rtx3090"],
        gpu_count=1,
        disk_size=30,
        model=dict(
            source_type="trainml",
            source_uri=model_id,
        ),
        environment=dict(
            type="DEEPLEARNING_PY39",
            env=env,
        ),
        endpoint=dict(
            start_command=start_command,
        ),
    )
    print(f"Created endpoint {job.id}, waiting for server startup")
    await job.wait_for("running", 1200)
    await asyncio.sleep(60)  ## model can take a minute to load into memory
    print(f"Endpoint running at  {job.url}")
    browse(job.url)


if __name__ == "__main__":
    args = parser.parse_args()
    trainml_client = TrainML()
    asyncio.run(
        launch_endpoint(
            trainml_client, args.model_id, args.server_type, args.model_type
        )
    )
