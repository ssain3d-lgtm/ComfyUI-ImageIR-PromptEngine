"""ComfyUI nodes that reach outside: vision backends, local servers, MiniMax H3.

These are the nodes that turn the package from a grounding library into a
pipeline. The intended graph is:

    Load Image -> Image IR Backend Config -> Image IR Analyzer -> IMAGE_IR
              -> MiniMax H3 Prompt Composer -> Image IR Grounding Guard

with Image IR Backend Control beside it when llama.cpp should be started from
ComfyUI rather than from a terminal.

Everything that knows how to talk to a model lives under imageir/backend, and
everything that knows how to read one's answer lives in imageir/analyzer. These
classes are the ComfyUI-shaped wrapper around them and hold no protocol
knowledge of their own — which is what lets the interesting parts be tested
without ComfyUI, a network, or a GPU.

On secrets: the api_token widget is the only place a key is typed, it is wrapped
before it leaves this file, and every debug output below is produced through the
masking helpers. A token should not be recoverable from any node output, any
error message, or any saved workflow.
"""

from __future__ import annotations

from .imageir.analyzer import AnalyzerError, analyze, load_system_prompt
from .imageir.backend import (
    PROVIDERS,
    SERVER_MODES,
    BackendConfig,
    BackendError,
    Secret,
    get_backend,
)
from .imageir.backend.llama_cpp_launcher import LAUNCHER, build_command
from .imageir.h3 import CAMERA_MOTIONS, FIELD_ORDER, compose_h3
from .imageir.imaging import DETAIL_MAX_SIDE, ImageError, image_to_payload
from .imageir.schema import IRError, parse_ir

CATEGORY = "IMAGE_IR"


def _as_ir(value):
    try:
        return parse_ir(value)
    except IRError as exc:
        raise ValueError(f"IMAGE_IR: {exc}") from None


class ImageIRBackendConfig:
    """Choose the vision backend once, and wire it wherever it is needed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (list(PROVIDERS), {
                    "default": "openai_compatible",
                    "tooltip": "llama_cpp: a llama-server, optionally launched from here. "
                               "openai_compatible: LM Studio, vLLM, or any server speaking the OpenAI chat API. "
                               "gemini: Google's hosted API.",
                }),
                "server_mode": (list(SERVER_MODES), {
                    "default": "connect_existing",
                    "tooltip": "launch_local starts llama-server from ComfyUI (llama_cpp only). "
                               "connect_existing talks to a server that is already running.",
                }),
                "model_name": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "The model id the server expects, e.g. gemma-3-12b-it or gemini-2.5-flash. "
                               "Not a file path — that is gguf_model_path.",
                }),
                "base_url": ("STRING", {
                    "default": "http://127.0.0.1:8080", "multiline": False,
                    "tooltip": "Server address for connect_existing. LM Studio defaults to "
                               "http://127.0.0.1:1234. Ignored when llama.cpp is launched here.",
                }),
                "api_token": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "AUTHENTICATION only — the API key. Required for Gemini, usually empty for a "
                               "local server. This is NOT max_tokens, which bounds the reply length.",
                }),
                "max_tokens": ("INT", {
                    "default": 2048, "min": 64, "max": 131072, "step": 64,
                    "tooltip": "Maximum LENGTH of the generated reply, in tokens. Unrelated to api_token. "
                               "IMAGE_IR for a detailed image needs room — 2048 is a reasonable floor.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Low is right here: this is a reading task, and creativity in a visual "
                               "description is indistinguishable from hallucination.",
                }),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Nucleus sampling cutoff."}),
                "timeout": ("FLOAT", {
                    "default": 120.0, "min": 5.0, "max": 1800.0, "step": 5.0,
                    "tooltip": "Seconds to wait for a reply. A local model loading for the first time can "
                               "take a while.",
                }),
            },
            "optional": {
                "llama_server_path": ("STRING", {
                    "default": "llama-server", "multiline": False,
                    "tooltip": "llama_cpp + launch_local: path to the llama-server binary, or its name if "
                               "it is on PATH.",
                }),
                "gguf_model_path": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "llama_cpp + launch_local: path to the .gguf weights.",
                }),
                "mmproj_path": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "llama_cpp + launch_local: the multimodal projector .gguf. Required for a "
                               "vision model; leave empty for text-only.",
                }),
                "host": ("STRING", {"default": "127.0.0.1", "multiline": False,
                                    "tooltip": "llama_cpp + launch_local: address to bind."}),
                "port": ("INT", {"default": 8080, "min": 1, "max": 65535,
                                 "tooltip": "llama_cpp + launch_local: port to bind."}),
                "context_size": ("INT", {
                    "default": 8192, "min": 512, "max": 1048576, "step": 512,
                    "tooltip": "llama_cpp + launch_local: --ctx-size. An image consumes a lot of it.",
                }),
                "gpu_layers": ("INT", {
                    "default": 0, "min": 0, "max": 1024,
                    "tooltip": "llama_cpp + launch_local: --n-gpu-layers. 0 is CPU only; a large number "
                               "offloads everything that fits.",
                }),
                "extra_args": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "llama_cpp + launch_local: extra llama-server flags, e.g. --jinja",
                }),
            },
        }

    RETURN_TYPES = ("IMAGEIR_BACKEND", "STRING")
    RETURN_NAMES = ("backend_config", "summary")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Pick the vision backend, model and generation settings. The API token never leaves this node in the clear."

    def run(
        self,
        provider="openai_compatible",
        server_mode="connect_existing",
        model_name="",
        base_url="http://127.0.0.1:8080",
        api_token="",
        max_tokens=2048,
        temperature=0.2,
        top_p=0.9,
        timeout=120.0,
        llama_server_path="llama-server",
        gguf_model_path="",
        mmproj_path="",
        host="127.0.0.1",
        port=8080,
        context_size=8192,
        gpu_layers=0,
        extra_args="",
    ):
        try:
            config = BackendConfig(
                provider=provider,
                server_mode=server_mode,
                base_url=base_url.strip(),
                api_token=Secret(api_token),
                model_name=model_name.strip(),
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                timeout=float(timeout),
                llama_server_path=llama_server_path.strip(),
                gguf_model_path=gguf_model_path.strip(),
                mmproj_path=mmproj_path.strip(),
                host=host.strip() or "127.0.0.1",
                port=int(port),
                context_size=int(context_size),
                gpu_layers=int(gpu_layers),
                extra_args=extra_args.strip(),
            )
        except BackendError as exc:
            raise ValueError(f"backend config: {exc}") from None

        lines = ["backend configuration", "=" * 21, ""]
        for key, value in config.redacted().items():
            lines.append(f"  {key:<18} {value}")
        if config.provider == "llama_cpp" and config.server_mode == "launch_local":
            lines.append("")
            lines.append("llama-server would be started as")
            lines.append("-" * 21)
            lines.append("  " + " ".join(build_command(config)))
            lines.append("")
            lines.append("  (the API token, if any, is passed in the environment, not on the")
            lines.append("   command line, so it cannot be read out of the process list)")
        return (config, "\n".join(lines))


class ImageIRBackendControl:
    """Start, stop and inspect a local llama-server, without touching anyone else's."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend_config": ("IMAGEIR_BACKEND",),
                "action": (["status", "start", "stop", "restart"], {
                    "default": "status",
                    "tooltip": "start never launches a second server on a port that already answers, and "
                               "stop only ever stops a process this extension started.",
                }),
                "ready_timeout": ("FLOAT", {
                    "default": 90.0, "min": 5.0, "max": 1800.0, "step": 5.0,
                    "tooltip": "Seconds to wait for the server to accept connections. Loading a large "
                               "model from a cold cache can take minutes.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGEIR_BACKEND", "STRING", "BOOLEAN")
    RETURN_NAMES = ("backend_config", "status", "running")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Launch or stop a local llama-server from ComfyUI. Never starts a duplicate, never stops a server it did not start."

    def run(self, backend_config, action="status", ready_timeout=90.0):
        config: BackendConfig = backend_config
        if config.provider != "llama_cpp":
            status = (
                f"provider is {config.provider}, which this extension does not launch.\n"
                f"There is nothing to start or stop — point base_url at the running server instead."
            )
            return (config, status, False)

        if action == "start":
            result = LAUNCHER.start(config, ready_timeout=float(ready_timeout))
        elif action == "stop":
            result = LAUNCHER.stop(config)
        elif action == "restart":
            result = LAUNCHER.restart(config, ready_timeout=float(ready_timeout))
        else:
            result = LAUNCHER.status(config)

        text = result.render()
        return (config, text, result.ok)


class ImageIRAnalyzer:
    """Read an image into IMAGE_IR with a vision model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "backend_config": ("IMAGEIR_BACKEND",),
                "analysis_detail": (list(DETAIL_MAX_SIDE), {
                    "default": "balanced",
                    "tooltip": "How much to look at, and at what resolution. fast reads the attributes that "
                               "decide likeness; detailed works every section and risk area.",
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Readings scoring below this are marked for verification and listed in "
                               "low_confidence_attributes. Nothing is deleted — the mark is for a future "
                               "verifier pass and for your own eyes.",
                }),
            },
            "optional": {
                "custom_system_prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Replaces prompts/image_ir_extractor.txt entirely. Leave empty to use the "
                               "shipped specification, which is where the risk-area guidance lives.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE_IR", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image_ir", "ir_json", "low_confidence_attributes", "compact_summary", "analyzer_debug_info")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Turn a ComfyUI IMAGE into IMAGE_IR using the configured vision backend, preserving every uncertainty it reports."

    def run(self, image, backend_config, analysis_detail="balanced", confidence_threshold=0.6, custom_system_prompt=""):
        config: BackendConfig = backend_config
        try:
            payload = image_to_payload(image, detail=analysis_detail)
        except ImageError as exc:
            raise ValueError(f"IMAGE: {exc}") from None

        backend = get_backend(config)
        try:
            result = analyze(
                backend,
                payload,
                detail=analysis_detail,
                confidence_threshold=float(confidence_threshold),
                system_prompt=custom_system_prompt,
            )
        except AnalyzerError as exc:
            # Already masked upstream; re-raised as ValueError so ComfyUI shows
            # it as a node error rather than a stack trace.
            raise ValueError(f"Image IR Analyzer: {exc}") from None

        debug = "\n".join(
            [
                "analyzer",
                "=" * 8,
                f"  provider    {config.provider}",
                f"  model       {result.model or config.model_name}",
                f"  endpoint    {backend.endpoint()}",
                f"  detail      {analysis_detail} (max side {DETAIL_MAX_SIDE[analysis_detail]}px)",
                f"  image       {len(payload.data)} bytes {payload.media_type}",
                f"  usage       {result.usage or '(not reported)'}",
                f"  attributes  {len(result.ir.facts)} across {len(result.ir.sections())} sections",
                "",
                "request (image elided, secrets masked)",
                "-" * 8,
                result.request_summary,
                "",
                "reply (secrets masked)",
                "-" * 8,
                result.raw_reply[:4000],
            ]
        )
        return (
            result.ir,
            result.ir.to_json(),
            result.low_confidence_report(float(confidence_threshold)),
            result.compact_summary(),
            debug,
        )


class MiniMaxH3PromptComposer:
    """Compose a MiniMax H3 I2VA prompt whose visual content is entirely IMAGE_IR's."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ir": ("IMAGE_IR",),
                "camera_motion": (list(CAMERA_MOTIONS), {
                    "default": "static",
                    "tooltip": "How the LENS moves. Kept separate from subject motion, and written into the "
                               "shot as an action rather than appended as a label, which is what the H3 "
                               "guide asks for.",
                }),
                "blink": ("BOOLEAN", {"default": False, "tooltip": "Subject motion: a natural blink."}),
                "breathing": ("BOOLEAN", {"default": False, "tooltip": "Subject motion: quiet breathing."}),
                "finger_movement": ("BOOLEAN", {"default": False, "tooltip": "Subject motion: the faintest finger movement."}),
                "hair_movement": ("BOOLEAN", {"default": False, "tooltip": "Subject motion: a few strands drifting."}),
                "weight_shift": ("BOOLEAN", {"default": False, "tooltip": "Subject motion: a small settling of weight."}),
                "require_motion_anchor": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On: a micro-motion is only used when IMAGE_IR records the feature it names.",
                }),
                "style": ("STRING", {
                    "default": "Live-action, cinematic", "multiline": False,
                    "tooltip": "Render style for the shot line, as the official examples carry it. Anything "
                               "here that describes the image rather than the rendering is caught by the audit.",
                }),
                "overall_soundscape": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Ambient and physical sound across the whole clip. A still frame records no "
                               "audio, so this is yours to write; empty becomes N/A.",
                }),
                "non_diegetic_music": ("STRING", {
                    "default": "N/A", "multiline": True,
                    "tooltip": "Score the characters cannot hear. N/A is a valid and common answer.",
                }),
                "on_violation": (["filter", "error", "report_only"], {
                    "default": "filter",
                    "tooltip": "What to do if the visual description fails its grounding audit.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("h3_prompt", "integrated_multimodal_description", "overall_soundscape", "non_diegetic_music", "trace", "audit")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Compose a MiniMax H3 I2VA prompt from IMAGE_IR, in the official field names and order."

    def run(
        self,
        image_ir,
        camera_motion="static",
        blink=False,
        breathing=False,
        finger_movement=False,
        hair_movement=False,
        weight_shift=False,
        require_motion_anchor=True,
        style="Live-action, cinematic",
        overall_soundscape="",
        non_diegetic_music="N/A",
        on_violation="filter",
    ):
        ir = _as_ir(image_ir)
        requested = [
            key
            for key, enabled in (
                ("blink", blink),
                ("breathing", breathing),
                ("finger_movement", finger_movement),
                ("hair_movement", hair_movement),
                ("weight_shift", weight_shift),
            )
            if enabled
        ]
        try:
            prompt = compose_h3(
                ir,
                motions=requested,
                require_anchor=require_motion_anchor,
                camera_motion=camera_motion,
                overall_soundscape=overall_soundscape,
                non_diegetic_music=non_diegetic_music,
                style=style,
                on_violation=on_violation,
            )
        except ValueError as exc:
            raise ValueError(f"MiniMax H3 composer: {exc}") from None

        report = prompt.audit_result.report() if prompt.audit_result else "no audit"
        return (
            prompt.render(),
            prompt.integrated_multimodal_description,
            prompt.overall_soundscape,
            prompt.non_diegetic_music,
            prompt.trace(),
            report,
        )


class ImageIRExtractorPrompt:
    """Show the shipped extractor specification, for reading or editing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Emit prompts/image_ir_extractor.txt, so it can be inspected or edited before feeding the analyzer."

    def run(self):
        return (load_system_prompt(),)


NODE_CLASS_MAPPINGS = {
    "ImageIRBackendConfig": ImageIRBackendConfig,
    "ImageIRBackendControl": ImageIRBackendControl,
    "ImageIRAnalyzer": ImageIRAnalyzer,
    "MiniMaxH3PromptComposer": MiniMaxH3PromptComposer,
    "ImageIRExtractorPrompt": ImageIRExtractorPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageIRBackendConfig": "Image IR Backend Config",
    "ImageIRBackendControl": "Image IR Backend Control",
    "ImageIRAnalyzer": "Image IR Analyzer",
    "MiniMaxH3PromptComposer": "MiniMax H3 Prompt Composer",
    "ImageIRExtractorPrompt": "Image IR Extractor Prompt",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "FIELD_ORDER"]
