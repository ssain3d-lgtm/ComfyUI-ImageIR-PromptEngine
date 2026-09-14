"""Local llama-server process management, without a llama-server.

Spawning and port probing are injected, so every branch that matters — the
duplicate that must not start, the foreign server that must not be stopped, the
binary that is not there — is exercised deterministically and in milliseconds.
"""

import unittest

import harness  # noqa: F401

from imageir.backend import BackendConfig, Secret
from imageir.backend.llama_cpp_launcher import (
    FAILED,
    RUNNING_EXTERNAL,
    RUNNING_OURS,
    STOPPED,
    LlamaServerLauncher,
    build_command,
    build_env,
)

TOKEN = "sk-test-fake"


class FakeProcess:
    """Just enough of Popen for the launcher to manage."""

    def __init__(self, pid=4321, exit_code=None):
        self.pid = pid
        self._exit_code = exit_code
        self.terminated = False
        self.killed = False
        self.stdout = None

    def poll(self):
        return self._exit_code

    def terminate(self):
        self.terminated = True
        self._exit_code = 0

    def kill(self):
        self.killed = True
        self._exit_code = -9

    def wait(self, timeout=None):
        return self._exit_code


def config(**kwargs):
    base = dict(
        provider="llama_cpp",
        server_mode="launch_local",
        llama_server_path="llama-server",
        gguf_model_path="/models/gemma.gguf",
        host="127.0.0.1",
        port=8080,
    )
    base.update(kwargs)
    return BackendConfig(**base)


def launcher(*, spawned=None, open_ports=()):
    """A launcher whose world is a list of open ports and a spawn recorder."""
    records = spawned if spawned is not None else []
    live = set(open_ports)

    def spawn(command, env):
        records.append((command, env))
        process = FakeProcess()
        live.add(8080 if "--port" not in command else int(command[command.index("--port") + 1]))
        return process

    instance = LlamaServerLauncher(spawn=spawn, probe=lambda host, port: port in live)
    instance.records = records
    instance.live = live
    # Path checks are about the user's disk, not about process handling; they
    # have their own tests below.
    instance._check_paths = lambda cfg: ""
    return instance


class CommandTests(unittest.TestCase):
    def test_the_current_long_form_flags_are_used(self):
        command = build_command(config(context_size=16384, gpu_layers=99, port=8081))
        self.assertEqual(command[0], "llama-server")
        for flag, value in (
            ("--model", "/models/gemma.gguf"),
            ("--host", "127.0.0.1"),
            ("--port", "8081"),
            ("--ctx-size", "16384"),
            ("--n-gpu-layers", "99"),
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, command)
                self.assertEqual(command[command.index(flag) + 1], value)

    def test_mmproj_is_included_only_when_given(self):
        self.assertNotIn("--mmproj", build_command(config()))
        command = build_command(config(mmproj_path="/models/mmproj.gguf"))
        self.assertEqual(command[command.index("--mmproj") + 1], "/models/mmproj.gguf")

    def test_extra_args_are_split_like_a_shell_would(self):
        command = build_command(config(extra_args='--jinja --alias "my model"'))
        self.assertIn("--jinja", command)
        self.assertIn("my model", command)

    def test_the_token_is_never_an_argument(self):
        # Anything on the command line is readable by every process on the box.
        command = build_command(config(api_token=Secret(TOKEN)))
        self.assertNotIn(TOKEN, " ".join(command))

    def test_the_token_travels_in_the_environment(self):
        env = build_env(config(api_token=Secret(TOKEN)), {})
        self.assertEqual(env["LLAMA_ARG_API_KEY"], TOKEN)

    def test_no_token_sets_no_variable(self):
        self.assertNotIn("LLAMA_ARG_API_KEY", build_env(config(), {}))


class StartTests(unittest.TestCase):
    def test_a_start_launches_once_and_reports_ready(self):
        control = launcher()
        status = control.start(config())
        self.assertEqual(status.state, RUNNING_OURS)
        self.assertEqual(len(control.records), 1)

    def test_a_second_start_does_not_launch_a_duplicate(self):
        # Two servers on one port is not a race the second wins; it is a crash,
        # or a model silently occupying memory for nothing.
        control = launcher()
        control.start(config())
        status = control.start(config())
        self.assertEqual(len(control.records), 1)
        self.assertEqual(status.state, RUNNING_OURS)
        self.assertIn("not starting a second", status.detail)

    def test_a_foreign_server_on_the_port_is_used_not_replaced(self):
        control = launcher(open_ports=(8080,))
        status = control.start(config())
        self.assertEqual(status.state, RUNNING_EXTERNAL)
        self.assertEqual(control.records, [])
        self.assertIn("connect_existing", status.detail)

    def test_a_process_that_dies_during_startup_reports_the_exit(self):
        control = LlamaServerLauncher(
            spawn=lambda command, env: FakeProcess(exit_code=1),
            probe=lambda host, port: False,
        )
        control._check_paths = lambda cfg: ""
        status = control.start(config(), ready_timeout=1.0)
        self.assertEqual(status.state, FAILED)
        self.assertIn("exited during startup", status.detail)

    def test_a_server_that_never_answers_times_out_rather_than_hanging(self):
        control = LlamaServerLauncher(
            spawn=lambda command, env: FakeProcess(),
            probe=lambda host, port: False,
        )
        control._check_paths = lambda cfg: ""
        status = control.start(config(), ready_timeout=0.5)
        self.assertEqual(status.state, FAILED)
        self.assertIn("did not answer", status.detail)

    def test_a_spawn_failure_does_not_escape(self):
        # An exception here would surface as a ComfyUI crash rather than a node error.
        def boom(command, env):
            raise OSError("Exec format error")

        control = LlamaServerLauncher(spawn=boom, probe=lambda host, port: False)
        control._check_paths = lambda cfg: ""
        status = control.start(config())
        self.assertEqual(status.state, FAILED)
        self.assertIn("Exec format error", status.detail)


class PathValidationTests(unittest.TestCase):
    def setUp(self):
        self.control = LlamaServerLauncher(spawn=lambda c, e: FakeProcess(), probe=lambda h, p: False)

    def test_a_missing_binary_is_named(self):
        status = self.control.start(config(llama_server_path="/nowhere/llama-server"))
        self.assertEqual(status.state, FAILED)
        self.assertIn("/nowhere/llama-server", status.detail)

    def test_an_empty_binary_path_is_rejected(self):
        self.assertIn("llama_server_path", self.control.start(config(llama_server_path="")).detail)

    def test_a_missing_gguf_is_named(self):
        control = LlamaServerLauncher(spawn=lambda c, e: FakeProcess(), probe=lambda h, p: False)
        control._check_paths = LlamaServerLauncher._check_paths.__get__(control)
        status = control.start(config(llama_server_path="python3", gguf_model_path="/nowhere/model.gguf"))
        self.assertIn("/nowhere/model.gguf", status.detail)

    def test_an_empty_gguf_path_is_rejected(self):
        status = self.control.start(config(llama_server_path="python3", gguf_model_path=""))
        self.assertIn("gguf_model_path", status.detail)


class StopTests(unittest.TestCase):
    def test_stopping_our_own_server_terminates_it(self):
        control = launcher()
        control.start(config())
        status = control.stop(config())
        self.assertEqual(status.state, STOPPED)
        self.assertFalse(control.owns("127.0.0.1", 8080))

    def test_a_server_we_did_not_start_is_left_alone(self):
        # The rule that matters most: port 8080 is very often the user's own
        # llama.cpp or LM Studio doing unrelated work.
        control = launcher(open_ports=(8080,))
        status = control.stop(config())
        self.assertEqual(status.state, RUNNING_EXTERNAL)
        self.assertIn("left alone", status.detail)

    def test_stopping_nothing_is_not_an_error(self):
        control = launcher()
        self.assertEqual(control.stop(config()).state, STOPPED)

    def test_stop_all_releases_only_what_we_own(self):
        control = launcher()
        control.start(config())
        control.stop_all()
        self.assertFalse(control.owns("127.0.0.1", 8080))

    def test_a_restart_refuses_when_the_port_is_someone_elses(self):
        control = launcher(open_ports=(8080,))
        self.assertEqual(control.restart(config()).state, RUNNING_EXTERNAL)
        self.assertEqual(control.records, [])


class StatusTests(unittest.TestCase):
    def test_status_distinguishes_ours_from_theirs_from_nothing(self):
        stopped = launcher()
        self.assertEqual(stopped.status(config()).state, STOPPED)

        external = launcher(open_ports=(8080,))
        self.assertEqual(external.status(config()).state, RUNNING_EXTERNAL)

        ours = launcher()
        ours.start(config())
        self.assertEqual(ours.status(config()).state, RUNNING_OURS)

    def test_the_rendered_status_never_leaks_the_token(self):
        control = launcher()
        control.start(config(api_token=Secret(TOKEN)))
        rendered = control.status(config(api_token=Secret(TOKEN))).render()
        self.assertNotIn(TOKEN, rendered)

    def test_the_rendered_status_names_the_url(self):
        self.assertIn("http://127.0.0.1:8080", launcher().status(config()).render())

    def test_ports_are_tracked_independently(self):
        control = launcher()
        control.start(config(port=8080))
        control.start(config(port=8081))
        self.assertEqual(len(control.records), 2)
        self.assertTrue(control.owns("127.0.0.1", 8080))
        self.assertTrue(control.owns("127.0.0.1", 8081))


if __name__ == "__main__":
    unittest.main()
