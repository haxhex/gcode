from dataclasses import dataclass
from typing import Dict, List, Any, Optional

@dataclass
class GoTConfig:
    config_id: str
    name: str
    n_init: int
    n_crit: int
    d_max: int
    p_fb: float
    n_path: int
    temperature: float
    archetype: str
    description: str
GOT_12_CONFIGS: Dict[str, GoTConfig] = {'C1': GoTConfig('C1', 'Narrow and Simple', 1, 1, 2, 0.0, 1, 0.2, 'Linear', 'Narrow and simple graph'), 'C2': GoTConfig('C2', 'Wide Multi-Idea', 5, 1, 2, 0.0, 1, 0.2, 'Wide', 'Wide graph with multiple initial ideas'), 'C3': GoTConfig('C3', 'Dense Critique', 3, 3, 2, 0.0, 1, 0.3, 'Critical', 'Dense critiques'), 'C4': GoTConfig('C4', 'Deep', 3, 1, 8, 0.0, 1, 0.2, 'Deep', 'Deep refinement'), 'C5': GoTConfig('C5', 'High Feedback', 3, 2, 5, 0.7, 1, 0.3, 'Feedback', 'High cross-feedback'), 'C6': GoTConfig('C6', 'Parallel Paths', 3, 1, 5, 0.3, 3, 0.4, 'Parallel', 'Parallel reasoning paths'), 'C7': GoTConfig('C7', 'High Diversity', 3, 2, 5, 0.3, 2, 0.1, 'Diverse', 'High graph diversity'), 'C8': GoTConfig('C8', 'Wide and Medium Depth', 5, 2, 4, 0.3, 2, 0.5, 'Wide', 'Wide graph with medium depth'), 'C9': GoTConfig('C9', 'Deep and Connected', 3, 2, 8, 0.3, 2, 0.5, 'Deep', 'Deep and connected graph'), 'C10': GoTConfig('C10', 'Feedback and Parallel', 3, 2, 5, 0.7, 3, 0.5, 'Parallel', 'Feedback with parallel paths'), 'C11': GoTConfig('C11', 'Wide Multi-Critique', 5, 3, 4, 0.3, 2, 0.5, 'Critical', 'Wide graph with multiple critiques'), 'C12': GoTConfig('C12', 'Varied Structure', 3, 3, 6, 0.5, 2, 0.7, 'Dense', 'Varied graph structure')}

@dataclass
class GoTGNNConfig:
    d_sem: int = 384
    d_abs: int = 5
    d_qual: int = 5
    d_type: int = 32
    d_proj: int = 128
    d_model: int = 256
    num_layers: int = 2
    num_heads: int = 4
    num_global_heads: int = 4
    d_cutoff: int = 4
    alpha: float = 1.0
    dropout: float = 0.2
    lr: float = 0.001
    weight_decay: float = 0.0001
    prune_threshold: float = 0.4
    path_success_threshold: float = 0.6
    thought_similarity_threshold: float = 0.75
    lambda_1: float = 0.25
    lambda_2: float = 0.25
    lambda_3: float = 0.25
    lambda_4: float = 0.25
    max_search_width: int = 8
    max_path_length: int = 6
    max_action_children: int = 5

@dataclass
class FlareGNNConfig:
    d_sem: int = 384
    d_stat_pre: int = 8
    d_stat_post: int = 14
    d_proj: int = 128
    d_flare: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.5
    lr: float = 0.001
    weight_decay: float = 0.0001
    gamma_focal: float = 2.0
    alpha_focal: float = 0.75
    lambda_rank: float = 0.5
    beta_target: float = 2.0
    lambda_stmt: float = 0.8
    lambda_method: float = 0.2
    lambda_mad: float = 1.5
    k_hop: int = 2
    b_max: int = 50

@dataclass
class PEMConfig:
    similarity_threshold: float = 0.75
    max_rules_per_cluster: int = 10
    decay_rate: float = 0.05
    min_confidence: float = 0.3
    top_k_retrieval: int = 3
    storage_path: Optional[str] = None

@dataclass
class PipelineConfig:
    max_iterations: int = 5
    default_planner_model: str = 'gpt-4o'
    default_generator_model: str = 'gpt-4o'
    dataset_name: str = 'rSDE-Bench'
    total_rsde_tasks: int = 53
    humaneval_tasks: int = 164