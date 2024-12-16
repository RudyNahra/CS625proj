import argparse
import json
import logging
import time
from multiprocessing import Process, Queue
from pathlib import Path

import torch
import torch.profiler
from torch.utils.data import DataLoader

from llama_3.args import ModelArgs
from llama_3.model_text_only import Transformer
from llama_3.tokenizer import Tokenizer
from openwebtext_sentences_dataset import OpenWebTextSentencesDataset
from wikitext2 import WikiText2Dataset
from utils.cuda_utils import set_up_cuda
import torch.nn.functional as F
from torchmetrics.text import Perplexity
from sae import TopKSparseAutoencoder, load_sae_model



def load_model(
    model_path: Path,
    model_args: ModelArgs,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    sae_layer_forward_fn: dict[int, callable] | None = None,
) -> Transformer:
    """Load and prepare the model."""
    # Initialize the model on CPU, usually with bfloat16 data type
    logging.info("Initializing model on CPU...")
    torch.set_default_dtype(dtype)
    model = Transformer(model_args, sae_layer_forward_fn=sae_layer_forward_fn) #Transformer(model_args, store_layer_activ=store_layer_activ)

    logging.info("Loading model weights into CPU memory...")
    model_weights = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )

    logging.info("Loading model weights into model...")
    model.load_state_dict(model_weights)
    del model_weights

    logging.info(f"Model weights loaded successfully. Moving model to device {device}...")
    model.to(device)

    logging.info("Setting model to eval mode...")
    model.eval()

    logging.info("Model created successfully.")
    return model


def compute_perplexity(
    model: Transformer,
    dataloader: DataLoader,
    device: torch.device,
    ignore_index: int,
    tokenizer: Tokenizer
) -> None:
    """Process batches and store activations."""
    # Create an update log every 0.5% through the process assigned batches
    #torch.set_default_dtype(torch.float64)
    state = {}
    with torch.no_grad():
        total_batches = len(dataloader)
        update_interval = max(1, total_batches // 200)

        pmetric = Perplexity(ignore_index=ignore_index).to(device).set_dtype(torch.float64)
        #logging.info(tokenizer.special_tokens)
        for batch_idx, (batch, indices, seq_lens) in enumerate(dataloader):
            # Move batch to device and set activation states to input
            torch.cuda.synchronize()

            targets = batch[:, 1:].contiguous().clone()
            
            #logging.info(f"padding index: {ignore_index}")
            #logging.info(tokenizer.decode(targets[0].tolist()))
            targets = targets.to(device)
            batch = batch.to(device)

            pad = torch.full((targets.size(0), 1), ignore_index, dtype=targets.dtype, device=targets.device)
            padded_targets = torch.cat((targets, pad), dim=1)

            
            logits = model(batch, start_pos=0)

            torch.cuda.synchronize()

            pmetric.update(logits, padded_targets)
            torch.cuda.synchronize()

            #logging.info(f"Total LL: {pmetric.total_log_probs}, {pmetric.total_log_probs.dtype} Num tokens: {pmetric.count}, {pmetric.count.dtype}")
            

            del targets
            del batch
            del logits
            torch.cuda.synchronize()

            # Update progress bar every 0.5% of the process assigned batches
            if (batch_idx + 1) % update_interval == 0:
                pmetric.state_dict(destination=state)
                progress = (batch_idx + 1) / total_batches
                logging.info(f"State: {state}")
                logging.info(f"Progress: {progress:.1%} ({batch_idx + 1}/{total_batches})\n PPL: {pmetric.compute()}")
                current_memory = torch.cuda.memory_allocated() / 1024**2  # in MB
                logging.info(f"Current GPU memory usage: {current_memory} MB")
                
        ppl = pmetric.compute()
        logging.info(f"Perplexity: {ppl}")
    #logging.info((prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10)))








def parse_arguments() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--drop", type=int, default=None) 
    parser.add_argument("--sae_dir", type=Path, default=None) 
    return parser.parse_args()


def main() -> None:
    """"""
   
    device = torch.device("cuda")
    set_up_cuda()



    # Parse arguments and set up paths
    args = parse_arguments()
    args.model_dir = args.model_dir.resolve()
    args.sae_dir = args.sae_dir.resolve()
    tokenizer_path = args.model_dir / "tokenizer.model"
    params_path = args.model_dir / "params.json"
    model_path = args.model_dir / "consolidated.00.pth"

    # Set up logging
    logname = "ppl.log"
    if args.drop is not None:
        logname = f"pplall{args.drop}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(logname),
            logging.StreamHandler()
        ],
    )



    # Set up configuration
    batch_size = 8
    dataloader_num_workers = 4
    dtype = torch.bfloat16
    max_token_length = 512
    add_bos_token = True
    dataset_shuffle = True

    sae_top_k = 64
    sae_normalization_eps = 1e-6
    batch_size = 8
    sae_layer_idx = 22
    sae_dtype = torch.float32
    sae_h_bias = None


    logging.info("#### Starting activation capture script.")
    logging.info("#### Arguments:")
    logging.info(f"# model_dir={args.model_dir}")
    logging.info(f"# sae_dir={args.sae_dir}")
    logging.info(f"# num_samples={args.num_samples}")
    logging.info(f"# drop latents={args.drop}")
    logging.info("#### GPU Configuration:")
    logging.info(f"# device={device}")
    logging.info("#### Configuration:")
    #logging.info(f"# store_layer_activ={store_layer_activ}")
    logging.info(f"# batch_size={batch_size}")
    logging.info(f"# dataloader_num_workers={dataloader_num_workers}")
    logging.info(f"# dtype={dtype}")
    logging.info(f"# max_token_length={max_token_length}")
    logging.info(f"# add_bos_token={add_bos_token}")
    logging.info(f"# dataset_shuffle={dataset_shuffle}")


    logging.info("Loading tokenizer...")
    tokenizer = Tokenizer(str(tokenizer_path))


    logging.info("Creating dataset, sampler and dataloader...")
    dataset = WikiText2Dataset(
        tokenizer=tokenizer,
        max_token_length=max_token_length,
        num_samples=args.num_samples,
        shuffle=dataset_shuffle,
        add_bos_token=add_bos_token,
        seed=43,
    )



    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )
    logging.info(f"Dataloader created with {len(dataloader)} batches.")
    mask_path = f"masks/mask_all_{args.drop}"
    logging.info(f"Loading activation mask {mask_path}")
    mask = torch.load(mask_path).to(device)

    logging.info(f"Initializing SAE from {args.sae_dir}")
    sae_layer_forward_fn = None
    if args.sae_dir is not None:
        assert sae_layer_idx is not None
        sae_model = load_sae_model(
            model_path=args.sae_dir,
            sae_top_k=sae_top_k,
            sae_normalization_eps=sae_normalization_eps,
            device=device,
            dtype=sae_dtype,
            activation_mask=mask,
        )
        sae_layer_forward_fn = {sae_layer_idx: sae_model.forward}

        if sae_h_bias is not None:
            logging.info("Setting SAE h_bias...")
            h_bias = torch.zeros(sae_model.n_latents)
            h_bias[sae_h_bias[0]] = sae_h_bias[1]
            h_bias = h_bias.to(sae_dtype).to(device)
            sae_model.set_latent_bias(h_bias)


    logging.info(f"Loading model parameters from {params_path}...")
    with params_path.open("r") as f:
        model_params = json.load(f)
    model_args = ModelArgs(**model_params)
    model = load_model(
        model_path=model_path,
        model_args=model_args,
        device=device,
        dtype=dtype,
        sae_layer_forward_fn=sae_layer_forward_fn,
    )


    logging.info("Starting capture of activations...")
    compute_perplexity(
        model=model,
        dataloader=dataloader,
        device=device,
        ignore_index=tokenizer.pad_id,
        tokenizer=tokenizer
    )

    # Wait for the saving process to finish

    logging.info("CUDA Memory Summary:")
    logging.info(torch.cuda.memory_summary())

    logging.info("FIN.")


if __name__ == "__main__":
    main()
