"""The RTX 5080 device: its published numbers, version gating, and that the simulator uses them."""

import json
import unittest

from mlsys_sim.agent_catalog import (CATALOG_VERSION, HARDWARE_BY_CATALOG_VERSION, get_agent_catalog,
                                     resolve_agent_device)
from mlsys_sim.agent_service import handle_agent_request
from mlsys_sim.schema import SimulationError

from test_agent_service import agent_fixture

V1, V2 = "mlsys-agent-catalog-v1", "mlsys-agent-catalog-v2"


def simulate(hardware, version=V2, input_tokens=2048, output_tokens=64):
    request = agent_fixture()
    request["catalog_version"] = version
    request["infra"]["nodes"][0]["devices"][0]["hardware"] = hardware
    request["workload"]["requests"] = [{"request_id": f"r{i}", "input_tokens": input_tokens,
                                        "output_tokens_target": output_tokens, "arrival_ms": i * 1000}
                                       for i in range(4)]
    request["strategy"]["deployments"][0]["model"] = "qwen3-1.7b"
    request["strategy"]["deployments"][0]["context_tokens"] = 4096
    request["strategy"]["serving"] = {"max_concurrency": 1, "continuous_batching": True, "kv_budget_gib": 1.0}
    request["evaluation"]["timeout_s"] = 30
    return handle_agent_request(json.dumps(request).encode())


def timings(response):
    """(first-token ms, decode ms per output token) of the first request."""
    row = response["requests"][0]
    return row["first_token_ms"], (row["finished_ms"] - row["first_token_ms"]) / (row["output_tokens"] - 1)


class RTX5080Test(unittest.TestCase):
    def test_nominal_numbers_are_the_vendor_dense_bf16_rate_with_a_source(self):
        device = resolve_agent_device("rtx5080", "node0/gpu0", V2)
        self.assertEqual(device.memory_bytes, 16 * 1024**3)
        self.assertEqual(device.memory_bandwidth_gbps, 960.0)  # 256-bit x 30 Gbps
        self.assertEqual(device.compute_tflops, 112.6)  # dense BF16 / FP32 accumulate, NOT the 225.1 sparse column
        entry = next(h for h in get_agent_catalog()["hardware"] if h["id"] == "rtx5080")
        self.assertEqual(entry["evidence"], "vendor_nominal")
        self.assertIn("nvidia-rtx-blackwell-gpu-architecture.pdf", entry["source"]["reference"])
        self.assertEqual(entry["since_catalog_version"], V2)

    def test_published_versions_never_lose_support(self):
        self.assertEqual(CATALOG_VERSION, V2)
        self.assertEqual(HARDWARE_BY_CATALOG_VERSION[V1], ("rx9070xt",))  # v1 is frozen
        self.assertEqual(resolve_agent_device("rx9070xt", "g", V1).compute_tflops, 195.0)

    def test_a_v1_request_cannot_reach_a_v2_device(self):
        with self.assertRaises(SimulationError) as caught:
            resolve_agent_device("rtx5080", "g", V1)
        self.assertEqual(caught.exception.code, "UNKNOWN_HARDWARE_ID")
        status, body = simulate("rtx5080", version=V1)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "UNKNOWN_HARDWARE_ID")

    def test_unknown_catalog_versions_are_still_refused(self):
        with self.assertRaises(SimulationError) as caught:
            resolve_agent_device("rx9070xt", "g", "mlsys-agent-catalog-v3")
        self.assertEqual(caught.exception.code, "CATALOG_VERSION_MISMATCH")

    def test_v1_results_do_not_change(self):
        """The same request under v1 and v2 on the same device is the same simulation."""
        (s1, r1), (s2, r2) = simulate("rx9070xt", version=V1), simulate("rx9070xt", version=V2)
        self.assertEqual((s1, s2), (200, 200))
        self.assertEqual(r1["requests"], r2["requests"])
        self.assertEqual((r1["catalog_version"], r2["catalog_version"]), (V1, V2))

    def test_the_simulator_uses_the_5080_rates_in_the_right_direction(self):
        """More bandwidth, less matrix compute: prefill (compute-bound) is slower, decode (bandwidth-bound) faster."""
        (s_amd, amd), (s_nv, nv) = simulate("rx9070xt"), simulate("rtx5080")
        self.assertEqual((s_amd, s_nv), (200, 200), (amd, nv))
        (amd_first, amd_decode), (nv_first, nv_decode) = timings(amd), timings(nv)
        self.assertGreater(nv_first, amd_first * 1.4)  # ~195/112.6 on a 2048-token prefill
        self.assertLess(nv_decode, amd_decode)  # 960 vs 640 GB/s on the weight reads


if __name__ == "__main__":
    unittest.main()
