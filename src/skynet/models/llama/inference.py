import time
import logging
import json
from email.policy import strict
from typing import Optional
from pathlib import Path

import torch
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer


LOG = logging.getLogger(__name__)


class LLaMa:
    """Inference of Large Language Model Meta AI

    :arg model:
    :arg tokenizer:
    :arg model_args:

    :type model:      `Transformer`
    :type tokenizer:  `SentencePieceProcessor`
    :type model_args: `ModelArgs`
    """
    def __init__(self, model, tokenizer, model_args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @classmethod
    def build(
            cls,
            max_batch_size,
            max_seq_len,
            params_fp=None,
            checkpoints_dir=None,
            tokenizer_path=None,
            load_model=False,
            device='cpu',
    ):
        """Building class method

        :param max_batch_size:
        :param max_seq_len:
        :param params_fp:
        :param checkpoints_dir:
        :param tokenizer_path:
        :param load_model:
        :param device:

        :type max_batch_size:   `int`
        :type max_seq_len:      `int`
        :type params_fp:        `str`
        :type checkpoints_dir:  `str`
        :type tokenizer_path:   `str`
        :type load_model:       `bool`
        :type device:           `str`|`torch.device`

        :rtype: `LLaMa`
        """
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            if len(checkpoints) == 0:
                raise FileNotFoundError("No checkpoints found.")
            chk_path = checkpoints[0]
            LOG.info("Loading checkpoint " + str(chk_path) + " ...")
            checkpoint = torch.load(chk_path, map_location='cpu')
            duration = time.time() - prev_time
            LOG.info("Loading checkpoint in " + str(duration) + "sec.")
            prev_time = time.time()

        params = {}
        if params_fp:
            with open(params_fp, 'r') as f:
                params = json.loads(f.read())
        model_args = ModelArgs(max_seq_len=max_seq_len,
                               max_batch_size=max_batch_size,
                               device=device,
                               **params)

        # Loading tokenizer
        tokenizer = SentencePieceProcessor()
        if tokenizer_path:
            tokenizer.Load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        # Setting the default tensor type according the device type:
        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        # Model initialization:
        model = Transformer(model_args)
        model = model.to(device)
        if load_model:
            del checkpoint['rope.freqs']  # noqa
            model.laod_state_dict(checkpoint, strict=True)
            duration = time.time() - prev_time
            LOG.info("Loaded state dict in " + str(duration) + " sec.")

        return cls(model, tokenizer, model_args)


def main():
    """Main function
    """
    torch.manual_seed(0)
    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    model = LLaMa.build(max_batch_size=3, max_seq_len=1024,
                        tokenizer_path="/home/mokira3d48/Documents/repositories/skynet/models/tokenizer.model",
                        device=device)
    print("Loading OK!")
    print("Model arguments: ", model.args)


if __name__ == '__main__':
    main()
