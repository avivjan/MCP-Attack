import json
import os
import random
import string
from typing import Dict, Any

from locust import HttpUser, task, between, events


# Add custom CLI flags
@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--scenario", choices=["handshake", "validate", "fanout", "mix"], default="mix")
    parser.add_argument("--kid-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--fixed-kid", default="kid-1")
    parser.add_argument("--jwt-bytes", type=int, default=512)
    parser.add_argument("--json-bytes", type=int, default=2048)
    parser.add_argument("--json-depth", type=int, default=5)
    parser.add_argument("--fanout-n", type=int, default=1)


def random_string(num: int) -> str:
    if num <= 0:
        return ""
    alphabet = string.ascii_letters + string.digits + ".-_"
    return "".join(random.choice(alphabet) for _ in range(num))


def make_fake_jwt(payload_bytes: int) -> str:
    # very rough JWT-like string: header.payload.signature
    header = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"  # base64 of {"alg":"HS256","typ":"JWT"}
    payload = random_string(max(1, payload_bytes))
    signature = random_string(43)
    return f"{header}.{payload}.{signature}"


def build_nested_json(target_bytes: int, depth: int) -> Dict[str, Any]:
    depth = max(1, depth)
    target_bytes = max(64, target_bytes)
    # Construct nested structure
    obj: Dict[str, Any] = {
        "user": "user-" + random_string(6),
        "action": "validate",
        "metadata": {
            "tags": ["perf", "test"],
            "nested": {
                "level": depth,
                "attributes": [
                    {"key": "k1", "value": 1},
                    {"key": "k2", "value": True},
                ],
            },
        },
    }
    # Inflate payload to approx target size
    blob = random_string(128)
    cur = obj
    for i in range(depth):
        cur["payload" if i == depth - 1 else f"lvl_{i}"] = {"blob": blob}
        cur = cur["payload" if i == depth - 1 else f"lvl_{i}"]
    # Size adjuster
    s = json.dumps(obj)
    deficit = target_bytes - len(s.encode("utf-8"))
    if deficit > 0:
        cur["pad"] = random_string(deficit)
    return obj


class MCPUser(HttpUser):
    wait_time = between(0.005, 0.02)

    def on_start(self):
        opts = self.environment.parsed_options
        self.scenario = opts.scenario
        self.kid_mode = opts.kid_mode
        self.fixed_kid = opts.fixed_kid
        self.jwt_bytes = opts.jwt_bytes
        self.json_bytes = opts.json_bytes
        self.json_depth = opts.json_depth
        self.fanout_n = opts.fanout_n

    def _pick_kid(self) -> str:
        if self.kid_mode == "fixed":
            return self.fixed_kid
        return f"kid-{random.randint(1, 10_000_000)}"

    def _handshake(self):
        token = make_fake_jwt(self.jwt_bytes)
        headers = {
            "Authorization": f"Bearer {token}",
            "X-KID": self._pick_kid(),
        }
        self.client.get("/handshake", headers=headers, name="handshake")

    def _validate(self):
        body = build_nested_json(self.json_bytes, self.json_depth)
        self.client.post(
            "/validate",
            data=json.dumps(body),
            headers={"Content-Type": "application/json"},
            name=f"validate_{self.json_bytes}B_d{self.json_depth}",
        )

    def _fanout(self):
        self.client.get(f"/fanout?n={self.fanout_n}", name=f"fanout_n={self.fanout_n}")

    @task
    def run_selected(self):
        if self.scenario == "handshake":
            self._handshake()
        elif self.scenario == "validate":
            self._validate()
        elif self.scenario == "fanout":
            self._fanout()
        else:  # mix
            choice = random.choice([self._handshake, self._validate, self._fanout])
            choice()
