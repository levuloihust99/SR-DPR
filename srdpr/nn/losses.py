import torch
import torch.nn.functional as F
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing import Text, Dict


class LossCalculator(object):
    def compute(
        self,
        inputs: Dict[Text, torch.Tensor],
        sim_func: Literal["cosine", "dot_product"],
        compute_type: Literal["inbatch", "pos_randneg", "pos_hardneg"],
        **kwargs
    ):
        if sim_func == "cosine":
            inputs = {k: self.normalize(v) for k, v in inputs.items()}
        if compute_type == 'pos_randneg':
            return self._compute_pos_randneg(
                question_embeddings=inputs['question_embeddings'],
                positive_context_embeddings=inputs['positive_context_embeddings'],
                negative_context_embeddings=inputs['negative_context_embeddings'],
                duplicate_mask=inputs['duplicate_mask'],
                **kwargs
            )
        elif compute_type == 'pos_hardneg':
            return self._compute_pos_hardneg(
                question_embeddings=inputs['question_embeddings'],
                positive_context_embeddings=inputs['positive_context_embeddings'],
                hardneg_context_embeddings=inputs['hardneg_context_embeddings'],
                hardneg_mask=inputs['hardneg_mask'],
                **kwargs
            )
        elif compute_type == "inbatch":
            raise Exception("Inbatch loss computation has not been implemented.")
        else:
            raise Exception("Unknown pipeline: '{}'".format(compute_type))
    
    def normalize(self, tensor: torch.Tensor):
        return tensor / F.normalize(tensor, dim=-1)
    
    def _compute_pos_randneg(
        self,
        question_embeddings: torch.Tensor,
        positive_context_embeddings: torch.Tensor,
        negative_context_embeddings: torch.Tensor,
        duplicate_mask: torch.Tensor
    ):
        batch_size, hidden_size = question_embeddings.size()
        positive_sim_scores = torch.sum(question_embeddings * positive_context_embeddings, dim=-1, keepdim=True)
        negative_sim_scores = torch.matmul(question_embeddings, torch.transpose(negative_context_embeddings, 0, 1))
        sim_matrix = torch.cat(
            [positive_sim_scores, negative_sim_scores],
            dim=-1
        )
        sim_matrix.masked_fill_(~duplicate_mask, -1e9)
        logits = F.log_softmax(sim_matrix, dim=-1)
        loss = logits[:, 0]
        loss = - torch.sum(loss) / batch_size
        return loss

    def _compute_pos_hardneg(
        self,
        question_embeddings: torch.Tensor,
        positive_context_embeddings: torch.Tensor,
        hardneg_context_embeddings: torch.Tensor,
        hardneg_mask: torch.Tensor
    ):
        batch_size, hidden_size = question_embeddings.size()
        positive_sim_scores = torch.sum(question_embeddings * positive_context_embeddings, dim=-1, keepdim=True)
        hardneg_sim_scores = torch.sum(question_embeddings.unsqueeze(1) * hardneg_context_embeddings, dim=-1)
        sim_matrix = torch.cat(
            [positive_sim_scores, hardneg_sim_scores],
            dim=-1
        )
        sim_matrix.masked_fill_(~hardneg_mask, -1e9)
        logits = F.log_softmax(sim_matrix, dim=-1)
        loss = logits[:, 0]
        loss = - torch.sum(loss) / batch_size
        return loss
