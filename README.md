# H²CLSR

## Running

Run from the project root, using a new `--runid` to avoid overwriting existing results:

```bash
# Ciao
python run_hclsr.py --dataset ciao --runid [new_id] --embedding_dim 64 --lr 0.003 --epochs 1000 --weight_decay 0.005 --num_layers 3 --negative_sampling random --interest_weight 0.8

# Epinions
python run_hclsr.py --dataset epinions --runid [new_id] --embedding_dim 64 --lr 0.003 --epochs 1000 --weight_decay 0.005 --num_layers 3 --negative_sampling random --interest_weight 0.8

# Yelp
python run_hclsr.py --dataset yelp --runid [new_id] --embedding_dim 64 --lr 0.003 --epochs 1000 --weight_decay 0.005 --num_layers 2 --negative_sampling random --interest_weight 0.8
```

Checkpoints and logs are saved under `saved/<dataset>/<runid>/`.

## Hyperparameters

Paper settings:

| Hyperparameter | Ciao | Epinions | Yelp |
| :--- | :---: | :---: | :---: |
| Embedding dimension $d$ | 64 | 64 | 64 |
| Learning rate | 0.003 | 0.003 | 0.003 |
| Training epochs | 1000 | 1000 | 1000 |
| Weight decay | 0.005 | 0.005 | 0.005 |
| Curvature | −1 | −1 | −1 |
| Temperature $\tau$ | 0.1 | 0.1 | 0.1 |
| Aggregation weight $\alpha_1$ | 0.8 | 0.8 | 0.8 |
| Aggregation weight $\alpha_2$ | 0.2 | 0.2 | 0.2 |
| Fusion weight $\beta$ | 0.8 | 0.8 | 0.8 |
| RLRA rank $k_1$ | 8 | 8 | 8 |
| Meta-network rank $k_2$ | 6 | 6 | 6 |
| Hard negative number $k_3$ | 80 | 80 | 80 |
| Graph propagation layers $L$ | 3 | 3 | 2 |
| Contrastive coefficient $\gamma_1$ | 0.6 | 0.6 | 0.6 |
| Contrastive coefficient $\gamma_2$ | 0.6 | 0.6 | 0.6 |
| Hierarchy-aware weighting coefficient $\lambda$ | 1 | 1 | 1 |
