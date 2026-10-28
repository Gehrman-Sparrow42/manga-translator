import threading
import unittest
from unittest.mock import Mock, patch

import requests

from utils.endpoints import openai_compatible as endpoint


class WaitHeartbeatTests(unittest.TestCase):
    def check_request(self, *, error: bool) -> None:
        seen = threading.Event()
        messages = []

        def capture(*args, **kwargs):
            messages.append((args, kwargs))
            seen.set()

        def post(*args, **kwargs):
            self.assertEqual(kwargs["timeout"], 23)
            self.assertTrue(seen.wait(2), "no heartbeat while the request was pending")
            if error:
                raise requests.exceptions.Timeout("test timeout")
            response = Mock()
            response.json.return_value = {
                "choices": [{"message": {"content": " ok "}}]
            }
            return response

        existing_threads = set(threading.enumerate())
        with (
            patch.object(endpoint, "_WAIT_HEARTBEAT_INTERVAL", 0.001),
            patch.object(endpoint.requests, "post", side_effect=post),
            patch.object(endpoint, "log_message"),
            patch("builtins.print", side_effect=capture),
        ):
            call = lambda: endpoint.call_openai_compatible_endpoint(
                "http://localhost/v1", None, "model", [{"text": "prompt"}], {},
                timeout=23, max_retries=0,
            )
            if error:
                with self.assertRaises(endpoint.TranslationError):
                    call()
            else:
                self.assertEqual(call(), "ok")

            count = len(messages)
            threading.Event().wait(0.02)
            self.assertEqual(len(messages), count)
            self.assertEqual(set(threading.enumerate()) - existing_threads, set())
            self.assertTrue(
                all(kwargs.get("flush") is True and "prompt" not in str(args)
                    for args, kwargs in messages)
            )

    def test_success_stops_heartbeat(self):
        self.check_request(error=False)

    def test_timeout_stops_heartbeat(self):
        self.check_request(error=True)


if __name__ == "__main__":
    unittest.main()
