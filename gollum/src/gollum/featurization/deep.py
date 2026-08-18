import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, List
from peft import LoraConfig, get_peft_model, TrainableTokensConfig
from peft.utils.peft_types import TaskType
from gollum.featurization.utils.pooling import average_pool, last_token_pool, weighted_average_pool
from gollum.featurization.text import get_model_and_tokenizer
from gollum.featurization.utils.layers import get_target_layers
from torch.nn import init
from tqdm import tqdm
import os
import gc
import contextlib

class BaseNNFeaturizer(nn.Module):
    """
    Base class for neural network-based featurizers.
    Combines nn.Module functionality with the BaseFeaturizer interface.
    
    This is specifically for featurizers that need neural network capabilities, such as LLM-based featurizers.
    """
    def __init__(
        self,
        input_dim: int = 768,
        projection_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
    
    @property
    def output_dim(self) -> int:
        """
        Returns the output dimension of the featurizer.
        
        Returns:
            int: Output dimension
        """
        return self._output_dim
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass through the neural network.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        pass

class ProjectionLayer(BaseNNFeaturizer):
    def __init__(
        self,
        input_dim: int = 3584,
        projection_dim: int = 64,
    ):
        super().__init__(input_dim=input_dim, projection_dim=projection_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(input_dim, projection_dim)

        self.fc1.bias.data.fill_(0.01)
        init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        dev = self.fc1.weight.device
        x = self.fc1(x)
        x = F.elu(x)
        if x.device != dev:
            x = x.to(dev, non_blocking=True)
        return x

class LLMFeaturizer(BaseNNFeaturizer):
    def __init__(
        self,
        model_name: str = "WhereIsAI/UAE-Large-V1",
        input_dim: int = 1024,
        projection_dim: Optional[int] = None,
        trainable: bool = True,
        pooling_method: str = "cls",
        normalize_embeddings: bool = False,
        lora_dropout: float = 0.2,
        modules_to_save: Optional[List[str]] = ["head"],
        target_ratio: float = 0.25,
        from_top: bool = True,
        tail_token_str: str = "<TT>",
        tail_token_id: Optional[int] = None, 
    ):
        super().__init__(input_dim=input_dim, projection_dim=projection_dim)
        print(model_name, "for LLM")
        self.llm, self.tokenizer = get_model_and_tokenizer(model_name, "cuda")

        self.tail_token_str = tail_token_str
        self.tail_token_id = tail_token_id
        
        if trainable:
            if self.tail_token_id is None:
                if self.tail_token_str not in self.tokenizer.get_vocab():
                    self.tokenizer.add_special_tokens(
                        {"additional_special_tokens": [self.tail_token_str]}
                    )
                self.tail_token_id = self.tokenizer.convert_tokens_to_ids(self.tail_token_str)

            self.llm.resize_token_embeddings(len(self.tokenizer))

            peft_cfg = TrainableTokensConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                token_indices=[self.tail_token_id],
                init_weights=True,
            )
            self.llm = get_peft_model(self.llm, peft_cfg)
            if hasattr(self.llm, "print_trainable_parameters"):
                self.llm.print_trainable_parameters()
        else:
            self.llm.requires_grad_(False)
            
        self.trainable = trainable
        self.embedding_dim = input_dim
        self.pooling_method = pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.input_dim = input_dim

        if projection_dim is not None:
            self.projector = ProjectionLayer(
                input_dim=input_dim, projection_dim=projection_dim
            )

        else:
            self.projector = nn.Identity()

        self.llm = self.llm.to(
            device=torch.device("cuda"), dtype=torch.bfloat16
        )
        self.projector = self.projector.to(
            device=torch.device("cuda"), dtype=torch.float32
        )

    def _apply(self, fn, *args, **kwargs):
        result = super()._apply(fn, *args, **kwargs)
        if getattr(self, "llm", None) is not None:
            self.llm.to(dtype=torch.bfloat16)
        return result

    def get_embeddings(self, x, batch_size=64, show_progress=False):

        n_points = x.size(0)
        ids_split = int(x.shape[-1] / 2)

        embedding_chunks = []

        for start_idx in tqdm(
            range(0, n_points, batch_size),
            total=(n_points + batch_size - 1) // batch_size,
            desc="Extracting token embeddings",
            disable=not show_progress,
            leave=False,
        ):

            end_idx = min(start_idx + batch_size, n_points)
            input_ids = x[start_idx:end_idx, :ids_split].long()
            attn_mask = x[start_idx:end_idx, ids_split:].long()

            real_len = int(attn_mask.sum(dim=1).max().item())
            input_ids = input_ids[:, :real_len]
            attn_mask = attn_mask[:, :real_len]

            grad_ctx = contextlib.nullcontext() if self.trainable else torch.no_grad()
            if not self.trainable:
                self.llm.eval()
            with grad_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.llm(
                    input_ids=input_ids, attention_mask=attn_mask,
                    use_cache=False,
                    output_hidden_states=False,
                )
                last_hidden_state = outputs.last_hidden_state

                if self.pooling_method == "average":
                    pooled = average_pool(last_hidden_state, attn_mask)
                elif self.pooling_method == "cls":
                    pooled = last_hidden_state[:, 0]
                elif self.pooling_method == "last_token_pool":
                    pooled = last_token_pool(last_hidden_state, attn_mask)
                elif self.pooling_method == "weighted_average":
                    pooled = weighted_average_pool(last_hidden_state, attn_mask)
                else:
                    raise ValueError(
                        f"Unknown pooling method: {self.pooling_method}"
                    )
            
            del outputs, last_hidden_state

            if self.normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=1)

            embedding_chunks.append(pooled.to(dtype=torch.float64))
            del pooled

        embeddings = torch.cat(embedding_chunks, dim=0)
        return embeddings

    def forward(self, x):

        if x.dim() == 3:

            n_candidates, n_train, d = x.shape
            train_data = x[0, : n_train - 1, :]
            all_candidates = x[:, n_train - 1, :]
            with torch.no_grad():

                train_embeddings = self.get_embeddings(train_data)
                all_candidate_embeddings = self.get_embeddings(all_candidates, show_progress=True)

            train_embeddings = train_embeddings.unsqueeze(0).expand(
                n_candidates, -1, -1
            )
            candidate_embeddings = all_candidate_embeddings.unsqueeze(1)
            embeddings = torch.cat(
                [train_embeddings, candidate_embeddings], dim=1
            )
            
            self.llm.to("cpu")
            del self.llm, self.tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        elif x.dim() == 2:
            embeddings = self.get_embeddings(x)

        return self.projector(embeddings)

    @property
    def output_dim(self):
        return (
            self.projector[-1].out_features
            if isinstance(self.projector, nn.Sequential)
            else self.embedding_dim
        )

