from typing import Dict, Any

import numpy as np
import bitshuffle


def map_from_uint16(data: np.ndarray, offset: float, scale: float) -> np.ndarray:
    dmax = np.iinfo(np.uint16).max
    # first scale, then offset:
    return (scale * data.astype(np.float32) / dmax) + offset


def map_to_uint16(data: np.ndarray) -> (np.ndarray, float, float):
    dmax = np.iinfo(np.uint16).max
    vmin = np.min(data)

    # map the beginning to zero:
    data = data - vmin

    # normalize to 0..1:
    new_max = np.max(data)
    if new_max != 0:
        data /= new_max

    # scale to the dtype range:
    data *= dmax

    return data, float(vmin), float(new_max)


class Codec:
    def encode(self, arr: np.ndarray) -> (bytes, Dict[str, Any]):
        """
        Encode `arr` into bytes and a `meta` dictionary.
        The returned `dict` should be json-serializable and
        contain the information needed to decode the `bytes` back
        into an array using the `decode` method.
        """
        raise NotImplementedError()

    def decode(self, encoded: bytes, meta: Dict[str, Any]) -> np.ndarray:
        """
        Decode the `encoded` bytes into an array, taking into account
        the `meta` dictionary which may contain helpful metadata.
        """
        raise NotImplementedError()


class LossyU16(Codec):
    def encode(self, arr: np.ndarray) -> (bytes, Dict[str, Any]):
        mapped_arr, offset, scale = map_to_uint16(arr)
        mapped_arr = mapped_arr.astype(np.uint16)
        encoding_meta = {
            'offset': offset,
            'scale': scale,
            'shape': tuple(arr.shape),
        }
        arr_compressed = bitshuffle.compress_lz4(mapped_arr)
        return arr_compressed, encoding_meta

    def decode(self, encoded: bytes, meta: Dict[str, Any]) -> np.ndarray:
        encoded = np.frombuffer(encoded, dtype=np.uint8)
        arr_uint16 = bitshuffle.decompress_lz4(
            encoded,
            dtype=np.dtype(np.uint16),
            shape=meta["shape"],
        )
        return map_from_uint16(arr_uint16, meta["offset"], meta["scale"])


class BsLz4(Codec):
    def encode(self, arr: np.ndarray) -> (bytes, Dict[str, Any]):
        if not arr.flags.c_contiguous:
            arr = np.copy(arr)
        arr_compressed = bitshuffle.compress_lz4(arr)
        return arr_compressed, {
            'dtype': str(arr.dtype),
            'shape': tuple(arr.shape),
        }

    def decode(self, encoded: bytes, meta: Dict[str, Any]) -> np.ndarray:
        encoded = np.frombuffer(encoded, dtype=np.uint8)
        return bitshuffle.decompress_lz4(
            encoded,
            dtype=np.dtype(meta['dtype']),
            shape=meta['shape'],
        )


try:
    import pytest  # NOQA

    def test_map_to_u16_simple():
        data = np.array([-1.0], dtype=np.float32)
        res, offset, scale = map_to_uint16(data)
        assert np.allclose(res, 0)
        assert offset == -1.0

    def test_map_to_u16_roundtrip_1():
        data = np.array([-1.0, 2, 65539], dtype=np.float32)
        res, offset, scale = map_to_uint16(data)
        assert res[2] == np.iinfo(np.uint16).max
        assert res[0] == np.iinfo(np.uint16).min
        assert np.allclose(
            data,
            map_from_uint16(res, offset, scale),
        )

    def test_map_to_u16_roundtrip_2():
        data = np.array([-1.0, -2.0], dtype=np.float32)
        res, offset, scale = map_to_uint16(data)
        assert np.allclose(
            data,
            map_from_uint16(res, offset, scale),
        )

    def test_codec_roundtrip_lossy_u16():
        # this fits into the range, so should be allclose:
        data = np.array([0, 2, 65535], dtype=np.float32)
        codec = LossyU16()
        compressed, meta = codec.encode(data)
        result = codec.decode(compressed, meta)
        assert np.allclose(result, data)

    def test_codec_roundtrip_bslz4():
        data = np.array([0, 2, 65535], dtype=np.float32)
        codec = BsLz4()
        compressed, meta = codec.encode(data)
        result = codec.decode(compressed, meta)
        assert np.allclose(result, data)

except ImportError:
    pass
