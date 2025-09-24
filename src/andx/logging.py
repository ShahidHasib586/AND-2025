from __future__ import annotations
from pathlib import Path
from typing import Optional
from rich.console import Console
from torch.utils.tensorboard import SummaryWriter

def get_logger(save_dir: Path, use_tb: bool = True):
    save_dir.mkdir(parents=True, exist_ok=True)
    console = Console()
    writer: Optional[SummaryWriter] = SummaryWriter(str(save_dir)) if use_tb else None
    return console, writer
