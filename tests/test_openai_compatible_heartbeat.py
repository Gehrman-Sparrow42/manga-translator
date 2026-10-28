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

class OllamaStreamingTests(unittest.TestCase):
    def invoke(self, response, url="http://localhost:11434/v1"):
        with patch.object(endpoint.requests, "post", return_value=response) as post, patch.object(endpoint, "log_message"):
            try:
                result = endpoint.call_openai_compatible_endpoint(
                    url, None, "model", [{"text": "prompt"}], {}, timeout=23, max_retries=0
                )
                return result, post.call_args.kwargs
            finally:
                if ":11434/" in url:
                    response.close.assert_called_once()

    def response(self, lines):
        response = Mock()
        response.headers = {"Content-Type": "text/event-stream"}
        response.iter_lines.return_value = iter(lines)
        return response

    def test_chunks_and_done(self):
        response = self.response([
            'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":" he"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"llo "},"finish_reason":"stop"}]}',
            'data: [DONE]',
        ])
        result, kwargs = self.invoke(response)
        self.assertEqual(result, "hello")
        self.assertTrue(kwargs["stream"])
        self.assertTrue(kwargs["json"]["stream"])
        self.assertEqual(kwargs["timeout"], 23)

    def test_reasoning_and_finish_without_done(self):
        response = self.response(['data: {"choices":[{"delta":{"reasoning":"answer"},"finish_reason":"stop"}]}'])
        self.assertEqual(self.invoke(response)[0], "answer")

    def test_error_malformed_and_truncated(self):
        for lines in ([ 'data: {"error":{"message":"test error"}}' ],
                      [ 'data: invalid' ],
                      [ 'data: {"choices":[{"delta":{"content":"partial"}}]}' ],
                      [ 'data: [DONE]' ]):
            with self.subTest(lines=lines), self.assertRaises(endpoint.TranslationError):
                self.invoke(self.response(lines))

    def test_json_fallback(self):
        response = Mock()
        response.headers = {"Content-Type": "application/json; charset=utf-8"}
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        self.assertEqual(self.invoke(response)[0], "ok")

    def test_non_ollama_unchanged(self):
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        result, kwargs = self.invoke(response, "http://localhost:8080/v1")
        self.assertEqual(result, "ok")
        self.assertNotIn("stream", kwargs)
        self.assertNotIn("stream", kwargs["json"])
        response.close.assert_not_called()

    def test_heartbeat_during_stream_and_cleanup_on_read_error(self):
        seen = threading.Event()
        response = self.response([])
        def lines(**kwargs):
            self.assertTrue(seen.wait(2))
            raise requests.exceptions.Timeout("test")
            yield
        response.iter_lines.side_effect = lines
        with (
            patch.object(endpoint, "_WAIT_HEARTBEAT_INTERVAL", 0.001),
            patch("builtins.print", side_effect=lambda *a, **k: seen.set()),
            self.assertRaises(endpoint.TranslationError),
        ):
            self.invoke(response)
        self.assertFalse(any(t.name.startswith("Thread-") and t.is_alive() for t in threading.enumerate()))
