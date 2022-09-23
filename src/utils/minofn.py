import numpy as np

class MinofN():
    @staticmethod
    def expand_modes(tensor, num_modes):
        batch_size = tensor.shape[0]
        expanded_tensor = tensor.unsqueeze(1).expand(
            batch_size, num_modes, *tensor.shape[1:])
        return expanded_tensor

    @staticmethod
    def repeat_modes(tensor, num_modes):
        ones_list = np.ones_like(tensor.shape[1:]).tolist()

        repeat_tensor = tensor.unsqueeze(1).repeat(
            1, num_modes, *ones_list)
        return repeat_tensor

    @staticmethod
    def compress_modes_into_batch(tensor):
        assert len(tensor.shape) >= 2, "Tensor must have at least batch x num_modes"
        tensor_compressed = tensor.reshape(-1, *tensor.shape[2:])
        return tensor_compressed

    @staticmethod
    def decompress_modes_from_batch(tensor, batch_size, num_modes):
        assert tensor.shape[0] == batch_size * num_modes, "Dim 0 != batch_size * num_modes"
        tensor_decompressed = tensor.reshape(
            batch_size, num_modes, *tensor.shape[1:])
        return tensor_decompressed

    @staticmethod
    def expand_and_compress_modes(tensor, num_modes):
        expanded_tensor = MinofN.expand_modes(tensor, num_modes)
        compressed_tensor = MinofN.compress_modes_into_batch(expanded_tensor)
        return compressed_tensor



