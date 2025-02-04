# GNN-DT: Graph Neural Network Enhanced Decision Transformer

## Introduction
GNN-DT (Graph Neural Network Enhanced Decision Transformer) is a novel decision-making framework that integrates Graph Neural Networks (GNNs) with Decision Transformers (DTs) to improve optimization in dynamic environments. Traditional reinforcement learning (RL) approaches often struggle with scalability, sparse rewards, and adaptability to changing state-action spaces. GNN-DT addresses these challenges by leveraging the permutation-equivariant properties of GNNs and a novel residual connection mechanism that ensures robust generalization across diverse scenarios.

This repository contains the implementation of GNN-DT, including dataset generation, model training, and evaluation scripts. The framework is designed for solving complex optimization problems, such as electric vehicle (EV) charging optimization, where efficiency and adaptability are crucial.

![image](https://github.com/user-attachments/assets/6de6459b-e681-4f7e-ac2f-10236979b109)


---

## Main Advantages
### 1. **Enhanced Sample Efficiency**
- Learns from previously collected trajectories, reducing the need for extensive online interactions.
- Effectively addresses the sparse rewards limitation of traditional RL algorithms.

### 2. **Robust Generalization**
- GNN-based embeddings allow for effective adaptation to unseen environments.
- Handles dynamic state-action spaces with varying numbers of entities over time.

### 3. **Superior Performance**
- Outperforms standard DT and RL baselines on real-world optimization tasks.
- Requires significantly fewer training trajectories while achieving higher rewards.

### 4. **Scalability**
- Maintains performance across different problem sizes without retraining.
- Efficiently scales from small-scale to large-scale environments, as demonstrated in EV charging applications.

![image](https://github.com/user-attachments/assets/a828ac07-564c-4ca7-889e-0a95fecd3689)

![image](https://github.com/user-attachments/assets/59c33604-e27e-43db-bfdf-2dc6c19cf914)



---

## Citation
If you find this repository useful in your research, please cite our paper:

```
@article{gnn-dt2025,
  title={GNN-DT: Graph Neural Network Enhanced Decision Transformer for Efficient Optimization in Dynamic Environments},
  author={},
  journal={},
  year={2025}
}
```

For any inquiries or contributions, feel free to open an issue or submit a pull request.


