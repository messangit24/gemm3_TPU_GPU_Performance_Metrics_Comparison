import os
import sys
import signal
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("coordinator_address", "",
                    "IP:port of JAX coordinator (unused - GKE auto-configures)")
flags.DEFINE_integer("num_processes", 2, "Total JAX processes (unused - GKE auto-configures)")
flags.DEFINE_integer("process_id", 0, "JAX process ID (unused - GKE auto-configures)")
flags.DEFINE_integer("port", 9000, "gRPC serving port")
flags.DEFINE_integer("threads", 64, "Server threads")
flags.DEFINE_bool("enable_model_warmup", False, "Enable model warmup")


def main_fn(argv):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    print("=" * 60, flush=True)
    print("JetStream Multihost Entrypoint (v2 - no manual init)", flush=True)
    print(f"  port        : {FLAGS.port}", flush=True)
    print(f"  threads     : {FLAGS.threads}", flush=True)
    print(f"  model_id    : {os.environ.get('JETSTREAM_MODEL_ID', 'NOT SET')}", flush=True)
    print(f"  TPU_WORKER_ID          : {os.environ.get('TPU_WORKER_ID', 'N/A')}", flush=True)
    print(f"  TPU_WORKER_HOSTNAMES   : {os.environ.get('TPU_WORKER_HOSTNAMES', 'N/A')}", flush=True)
    print(f"  TPU_PROCESS_ADDRESSES  : {os.environ.get('TPU_PROCESS_ADDRESSES', 'N/A')}", flush=True)
    print("=" * 60, flush=True)

    # GKE sets TPU_WORKER_HOSTNAMES, TPU_WORKER_ID, TPU_PROCESS_ADDRESSES
    # JAX uses these to auto-initialize distributed mode
    # Do NOT call jax.distributed.initialize() manually

    import jax
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")

    print(f"JAX ready!", flush=True)
    print(f"  device_count     : {jax.device_count()}", flush=True)
    print(f"  local_device_count : {jax.local_device_count()}", flush=True)
    print(f"  process_index    : {jax.process_index()}", flush=True)
    print(f"  process_count    : {jax.process_count()}", flush=True)

    import torch
    import torch_xla2
    from jetstream_pt import fetch_models, environment, engine
    from jetstream_pt import quantize_model, torchjax, config
    from jetstream.core import server_lib
    from jetstream.core.config_lib import ServerConfig
    from transformers import AutoTokenizer

    def shard_weights(env, weights, weight_shardings):
        sharded = {}
        for key, val in weights.items():
            sharding = env.sharding_by_axis(weight_shardings.get(key, -1))
            with jax.default_device(jax.devices("cpu")[0]):
                arr = torch_xla2.tensor.t2j(val)
            arr = jax.device_put(arr, sharding)
            sharded[key] = torchjax.to_torch(arr)
        return sharded

    def create_engine(devices):
        model_id = os.environ["JETSTREAM_MODEL_ID"]
        batch_size = int(os.environ.get("JETSTREAM_BATCH_SIZE", "8"))
        max_input = int(os.environ.get("JETSTREAM_MAX_INPUT", "2048"))
        max_output = int(os.environ.get("JETSTREAM_MAX_OUTPUT", "1024"))

        print(f"create_engine: model_id={model_id}", flush=True)
        print(f"create_engine: batch={batch_size} max_in={max_input} max_out={max_output}", flush=True)

        torch.set_default_dtype(torch.bfloat16)
        quant_config = config.create_quantization_config_from_flags()
        config.set_jax_compilation_cache_config()
        env_data = fetch_models.construct_env_data_from_model_id(
            model_id, batch_size, max_input, max_output,
        )
        env = environment.JetEngineEnvironment(env_data)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        env.hf_tokenizer = tokenizer
        model = fetch_models.instantiate_model_from_repo_id(model_id, env)
        env.quant_config = quant_config
        if quant_config.enable_weight_quantization:
            quantize_model.quantize_model(model, quant_config)
        weight_shardings = model.get_sharding_annotations()
        sharded_weights = shard_weights(env, model.state_dict(), weight_shardings)
        env_data.quant_config = quant_config
        return engine.PyTorchEngine(
            pt_model=model,
            env=env,
            weights=torchjax.from_torch_with_copy(sharded_weights),
        )

    if jax.process_index() == 0:
        print("==> Leader: starting JetStream gRPC...", flush=True)
        devices = server_lib.get_devices()
        print(f"    Devices: {devices}", flush=True)
        server_config = ServerConfig(
            interleaved_slices=(f"tpu={len(jax.devices())}",),
            interleaved_engine_create_fns=[create_engine],
        )
        jetstream_server = server_lib.run(
            threads=FLAGS.threads,
            port=FLAGS.port,
            config=server_config,
            devices=devices,
            metrics_server_config=None,
            enable_model_warmup=FLAGS.enable_model_warmup,
        )
        print(f"JetStream running on port {FLAGS.port}", flush=True)
        jetstream_server.wait_for_termination()
    else:
        print("==> Worker: blocking for JAX collectives...", flush=True)
        signal.pause()


if __name__ == "__main__":
    app.run(main_fn)
