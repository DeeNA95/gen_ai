#! /usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import string
import os  # Import os for environment variables

# --- XLA Imports ---
# Keep these imports conditional
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.runtime as xr  # Import runtime

    _xla_available = True
except ImportError:
    _xla_available = False
# --- End XLA Imports ---

# --- Standard Parallelism Imports ---
from torch.nn.parallel import DataParallel

# --- End Standard Parallelism Imports ---

# Global flags/config (adjust as needed)
FLAGS = {
    'batch_size': 256,
    'block_size': 256,
    'max_iters': 2000,
    'eval_iters': 200,
    'n_embd': 384,
    'lr': 3e-4,
    'n_heads': 6,  # Renamed N_HEADS for consistency
    'dropout': 0.2,
    'n_layers': 6,
    'seed': 1337,  # Added for reproducibility
    'log_interval': 100,  # How often to print loss
}

# Set seed for reproducibility
torch.manual_seed(FLAGS['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(FLAGS['seed'])


# --- Data Loading ---
# Moved data loading outside the main function for potential sharing
def load_data(data_path='GPT/data/wikitext-2-raw'):
    train_path = os.path.join(data_path, 'wiki.train.raw')
    valid_path = os.path.join(data_path, 'wiki.valid.raw')

    with open(train_path, 'r') as f:
        text = f.read()
    with open(valid_path, 'r') as f:
        valid = f.read()

    print(f'First 1000 characters \n {text[:1000]}\n\n')
    return text, valid


def filter_english_chars(vocab_string):
    allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + ' ' + '\n')
    filtered_vocab = ''.join(c for c in vocab_string if c in allowed_chars)
    return filtered_vocab, allowed_chars


def filter_text_by_vocab(text_var, allowed_chars):
    filtered_text = ''.join(c for c in text_var if c in allowed_chars)
    return filtered_text


# Prepare vocabulary and tokenizers
raw_text, raw_valid = load_data()
combined_chars = sorted(list(set(raw_text + raw_valid)))
_, allowed_chars = filter_english_chars(''.join(combined_chars))
vocab_chars = sorted(list(allowed_chars))  # Use the allowed chars as vocab
vocab_size = len(vocab_chars)

text = filter_text_by_vocab(raw_text, allowed_chars)
valid = filter_text_by_vocab(raw_valid, allowed_chars)

s_to_i = {ch: i for i, ch in enumerate(vocab_chars)}
i_to_s = {i: ch for i, ch in enumerate(vocab_chars)}
encode = lambda s: [s_to_i[c] for c in s]
decode = lambda l: ''.join([i_to_s.get(i, '?') for i in l])  # Use get for safety

print(f'characters available in vocab ({vocab_size}): {"".join(vocab_chars)}')

# Tokenize data (move to CPU first, then transfer in get_batch/loader)
data = torch.tensor(encode(text), dtype=torch.long)
test = torch.tensor(encode(valid), dtype=torch.long)


# Dataloading function (works for CPU/GPU, adapted for XLA later)
def get_batch(split, device):
    batch_data = data if split == 'train' else test
    ix = torch.randint(len(batch_data) - FLAGS['block_size'], (FLAGS['batch_size'],))
    x = torch.stack([batch_data[i:i + FLAGS['block_size']] for i in ix])
    y = torch.stack([batch_data[i + 1:i + FLAGS['block_size'] + 1] for i in ix])
    # Move data to the target device within the function
    return x.to(device), y.to(device)


# --- Model Definition ---
# (Head, MultiHeadAttention, FeedForward, Block remain the same, ensure they use FLAGS['dropout'])
class Head(nn.Module):
    """single head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(FLAGS['n_embd'], head_size, bias=False)
        self.query = nn.Linear(FLAGS['n_embd'], head_size, bias=False)
        self.value = nn.Linear(FLAGS['n_embd'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(FLAGS['block_size'], FLAGS['block_size'])))
        self.dropout = nn.Dropout(FLAGS['dropout'])  # Use FLAGS

    def forward(self, x):
        B, T, C = x.shape  # C is n_embd
        k = self.key(x)  # B, T, head_size
        q = self.query(x)  # B, T, head_size
        v = self.value(x)  # B, T, head_size

        # Use head_size which is C // n_heads for scaling if input C is n_embd
        head_size = k.shape[-1]
        weights = (q @ k.transpose(-2, -1)) * (head_size ** -0.5)  # B x T x T
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # B x T x T
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v  # (B, T, T) @ (B, T, head_size) -> B, T, head_size
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        # Project concatenated heads (head_size * n_heads) back to n_embd
        self.proj = nn.Linear(head_size * n_heads, FLAGS['n_embd'])
        self.dropout = nn.Dropout(FLAGS['dropout'])  # Use FLAGS

    def forward(self, x):
        # Input x is (B, T, n_embd)
        # Each head outputs (B, T, head_size)
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # B, T, head_size * n_heads
        out = self.dropout(self.proj(out))  # Project back to B, T, n_embd
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # Changed from ReLU to GELU, common in transformers
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(FLAGS['dropout'])  # Use FLAGS
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(head_size, n_heads)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-LayerNorm variation (common in modern transformers)
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size_arg):  # Pass vocab_size
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size_arg, FLAGS['n_embd'])
        self.position_embedding_table = nn.Embedding(FLAGS['block_size'], FLAGS['n_embd'])
        self.blocks = nn.Sequential(
            *[Block(FLAGS['n_embd'], FLAGS['n_heads']) for _ in range(FLAGS['n_layers'])],
            nn.LayerNorm(FLAGS['n_embd'])  # Final LayerNorm
        )
        self.linear_layer = nn.Linear(FLAGS['n_embd'], vocab_size_arg)
        # Removed unused self.sa_head and self.ff from here

        # Weight tying (optional but common)
        self.token_embedding_table.weight = self.linear_layer.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)  # B x T x n_embd
        postion_embeddings = self.position_embedding_table(torch.arange(T, device=idx.device))  # T x n_embd
        x = token_embeddings + postion_embeddings  # B x T x n_embd
        x = self.blocks(x)  # Pass through transformer blocks

        logits = self.linear_layer(x)  # B x T x vocab_size

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_for_loss = logits.view(B * T, C)  # Reshape for cross-entropy
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits_for_loss, targets)

        return logits, loss

    @torch.no_grad()  # Ensure no gradients are computed during generation
    def generate(self, idx, max_new_tokens):
        self.eval()  # Set model to evaluation mode
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens if necessary
            idx_cond = idx[:, -FLAGS['block_size']:]
            # Get predictions
            logits, _ = self(idx_cond)  # We don't need loss here
            # Focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        self.train()  # Set model back to train mode
        return idx


# --- Loss Estimator ---
@torch.no_grad()
def estimate_loss(model, device, using_xla=False):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(FLAGS['eval_iters'])
        for k in range(FLAGS['eval_iters']):
            # Use standard get_batch for simplicity here, XLA loader used in main train loop
            x, y = get_batch(split, device)
            # No need for model(x,y) if model is DataParallel, it handles scattering
            logits, loss = model(x, y)

            # If using DataParallel, loss might be a tensor on each device.
            # Average across devices if necessary (DataParallel usually does this)
            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                losses[k] = loss.mean().item()  # Average if multiple losses returned
            elif isinstance(loss, torch.Tensor):
                losses[k] = loss.item()  # Single loss value
            else:  # Handle cases like XLA where loss might already be aggregated?
                losses[k] = loss  # Assume loss is already a scalar or handle appropriately

        # For XLA, aggregate losses across devices
        if using_xla:
            losses = xm.all_reduce(xm.REDUCE_SUM, losses) / xr.world_size()  # Use xr.world_size
            out[split] = losses.mean().item()  # Get scalar value after reduction
        else:
            out[split] = losses.mean().item()  # Get scalar value

    model.train()
    return out


# --- Main Training Function (for XLA multiprocessing) ---
def _mp_fn(rank, flags):
    global model  # Allow modification of global model if needed (careful)
    torch.manual_seed(flags['seed'])  # Seed each process

    device = xm.xla_device()
    print(f"Rank {rank}: Using TPU via PyTorch/XLA: {device}")

    # Create model and move it to the XLA device for this process
    model = Transformer(vocab_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=flags['lr'])

    # Create XLA-aware dataloader
    # Note: Requires a Dataset class, adapting get_batch for simplicity
    # A proper implementation would use torch.utils.data.Dataset
    # This is a simplified placeholder:
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, tensor_data, block_size):
            self.data = tensor_data
            self.block_size = block_size

        def __len__(self):
            # Adjust length to account for block_size
            return len(self.data) - self.block_size

        def __getitem__(self, idx):
            x = self.data[idx:idx + self.block_size]
            y = self.data[idx + 1:idx + self.block_size + 1]
            return x, y

    train_dataset = SimpleDataset(data, flags['block_size'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xr.world_size(),  # Use xr.world_size
        rank=rank,
        shuffle=True,
        seed=flags['seed']
    )
    # Use ParallelLoader
    train_loader = pl.ParallelLoader(train_dataset, [flags['batch_size']], sampler=train_sampler)
    train_device_loader = train_loader.per_device_loader(device)  # Get loader for this specific device

    # Training loop for XLA
    model.train()
    for iter_num in range(flags['max_iters']):
        # Get batch from the XLA loader for this device
        # This might require iteration: for x, y in train_device_loader:
        # Simplified: assuming loader yields batches directly
        try:
            x, y = next(iter(train_device_loader))  # Get next batch
        except StopIteration:
            # Reset loader if epoch ends (depends on loader implementation)
            train_device_loader = train_loader.per_device_loader(device)
            x, y = next(iter(train_device_loader))

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        xm.optimizer_step(optimizer)  # Use XLA optimizer step

        if iter_num % flags['log_interval'] == 0:
            # Loss estimation needs care in distributed setting
            # estimate_loss needs to be adapted or called only on rank 0
            # For simplicity, print local loss, but estimate_loss is better
            if rank == 0:  # Only rank 0 estimates and prints global loss
                losses = estimate_loss(model, device, using_xla=True)
                print(
                    f"Rank {rank} Iter {iter_num}: Train Loss {losses['train']:.4f}, Valid Loss {losses['val']:.4f} (Local Batch Loss: {loss.item():.4f})")
            else:
                # Other ranks just mark step
                xm.mark_step()  # Ensure graph execution progresses

    # --- Generation (only on rank 0) ---
    if rank == 0:
        print("Training complete on TPUs.")
        print("Generating text...")
        # Ensure model is on the correct device for generation if needed
        model.eval()  # Set to eval mode
        start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_indices = model.generate(start_context, max_new_tokens=500)[0].tolist()
        print(decode(generated_indices))


# --- Main Execution Logic ---
if __name__ == '__main__':
    # --- Device Selection ---
    device = None
    using_xla_multiprocessing = False

    if _xla_available:
        world_size = xr.world_size()  # Use xr.world_size
        if world_size > 1:
            print(f"Found {world_size} XLA devices. Using XLA multiprocessing.")
            # Spawn processes for XLA training
            xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=world_size, start_method='fork')
            using_xla_multiprocessing = True
        elif world_size == 1:
            device = xm.xla_device()
            print("Using single TPU via PyTorch/XLA")
        else:  # world_size = 0 or error
            print("XLA available but no devices found/configured.")

    if not using_xla_multiprocessing and device is None:  # If XLA not used or only 1 device found
        if torch.cuda.is_available():
            device = torch.device("cuda")
            n_gpu = torch.cuda.device_count()
            print(f"Using {n_gpu} CUDA GPU(s)")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Metal (MPS)")
        else:
            device = torch.device("cpu")
            print("Using CPU")

    print(f'Selected device: {device}')  # Will be None if XLA multiprocessing started

    # --- Run Training (if not using XLA multiprocessing) ---
    if not using_xla_multiprocessing:
        # Initialize model
        model = Transformer(vocab_size).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

        # Check for multiple GPUs and wrap with DataParallel if needed
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            print(f"Wrapping model with DataParallel for {torch.cuda.device_count()} GPUs.")
            model = DataParallel(model)  # Wrap the model

        # init optimiser
        # If using DataParallel, optimizer works on model.module.parameters() or just model.parameters()
        optimizer = optim.AdamW(model.parameters(), lr=FLAGS['lr'])

        # Training loop (non-XLA)
        model.train()
        for iter_num in range(FLAGS['max_iters']):

            if iter_num % FLAGS['log_interval'] == 0:
                losses = estimate_loss(model, device)  # Pass model and device
                print(f"Iter {iter_num}: Train Loss {losses['train']:.4f}, Valid Loss {losses['val']:.4f}")

            x, y = get_batch('train', device)  # Get batch for the target device
            logits, loss = model(x, y)

            # If using DataParallel, loss is automatically averaged across GPUs
            # Need to handle the case where loss might be a tensor with loss per GPU
            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                final_loss = loss.mean()  # Average losses if multiple returned
            else:
                final_loss = loss  # Use the single loss value

            optimizer.zero_grad(set_to_none=True)
            final_loss.backward()  # Backpropagate the averaged loss
            optimizer.step()

        print("Training complete.")

        # --- Generation (non-XLA) ---
        print("Generating text...")
        # If using DataParallel, access the original model via model.module
        model_to_generate = model.module if isinstance(model, DataParallel) else model
        model_to_generate.eval()  # Set to eval

        start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_indices = model_to_generate.generate(start_context, max_new_tokens=500)[0].tolist()
        print(decode(generated_indices))
