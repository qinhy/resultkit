from __future__ import annotations

import ctypes

MAX_JSONRPC_PAYLOAD = 64 * 1024


class JsonRpcPacket(ctypes.Structure):
    """Fixed-size shared-memory-compatible packet for JSON bytes."""

    _pack_ = 1
    _fields_ = [
        ("correlation_id", ctypes.c_uint64),
        ("payload_len", ctypes.c_uint32),
        ("payload", ctypes.c_uint8 * MAX_JSONRPC_PAYLOAD),
    ]


class JsonRpcPacketCodec:
    def __init__(self, max_payload_bytes: int = MAX_JSONRPC_PAYLOAD) -> None:
        if max_payload_bytes <= 0 or max_payload_bytes > MAX_JSONRPC_PAYLOAD:
            raise ValueError(f"max_payload_bytes must be between 1 and {MAX_JSONRPC_PAYLOAD}")
        self.max_payload_bytes = max_payload_bytes

    def encode(self, payload: bytes, *, correlation_id: int) -> JsonRpcPacket:
        if len(payload) > self.max_payload_bytes:
            raise ValueError(f"JSON-RPC payload is too large: {len(payload)} > {self.max_payload_bytes}")

        packet = JsonRpcPacket()
        packet.correlation_id = correlation_id
        packet.payload_len = len(payload)
        for index, byte in enumerate(payload):
            packet.payload[index] = byte
        return packet

    def decode(self, packet: JsonRpcPacket) -> bytes:
        payload_len = int(packet.payload_len)
        if payload_len > self.max_payload_bytes:
            raise ValueError(f"Invalid JSON-RPC packet length: {payload_len} > {self.max_payload_bytes}")
        return bytes(packet.payload[:payload_len])
