# Reference audit for the NeurIPS manuscript

Audit date: **2026-09-01**. Every entry in `paper/references.bib` was checked
against a primary source: an official proceedings page, OpenReview record,
arXiv record, author-hosted paper record, or official project page. The
bibliography intentionally excludes unverified secondary-source metadata.

## Status conventions

- **Proceedings** means a final conference record was verified and is cited in
  preference to the corresponding arXiv preprint.
- **arXiv preprint** means no final archival venue was asserted by the primary
  source at audit time. The bibliography does not invent a venue.
- An arXiv-issued DOI (`10.48550/arXiv...`) identifies the preprint; it does not
  imply conference acceptance.

## Verification and citation-purpose map

| Key | Verified publication metadata and primary source | Intended manuscript use |
|---|---|---|
| `zhang2026rift` | Chushan Zhang, Jinguang Tong, Xuesong Li, Yikai Wang, Hongdong Li. *Keep the Future, Drop the Rollout: RIFT for World Action Models*. arXiv:2608.11521v2, submitted 12 Aug 2026, revised 13 Aug 2026. [arXiv](https://arxiv.org/abs/2608.11521) | Closest related work. RIFT establishes causal dependence on future-position K/V values and positional organization through masking, corruption, and reassignment, and shows final-clean-cache replay for supported architectures. Contrast its unsigned execution divergence/necessity tests with our signed, coherent-donor semantic steering estimand. Do **not** say RIFT only measures performance or does not affect trajectories. |
| `kim2026cosmospolicy` | Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, Jinwei Gu. *Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning*. arXiv:2601.16163, 22 Jan 2026. [arXiv](https://arxiv.org/abs/2601.16163), [official project](https://research.nvidia.com/labs/dir/cosmos-policy/) | Architecture/checkpoint provenance for the Predict2-based experiments and the original model's planning claims. |
| `nvidia2026cosmos3` | NVIDIA. *Cosmos 3: Omnimodal World Models for Physical AI*. arXiv:2606.02800v4, submitted 1 Jun 2026, revised 23 Jun 2026. [arXiv](https://arxiv.org/abs/2606.02800), [official project and requested BibTeX](https://research.nvidia.com/labs/cosmos-lab/cosmos3/) | Architecture and checkpoint-family provenance for the Cosmos 3 replication. NVIDIA explicitly asks that the report be cited with corporate author `NVIDIA`; the BibTeX follows that instruction rather than reproducing the 294-person contributor appendix. |
| `yuan2026fastwam` | Tianyuan Yuan, Zibin Dong, Yicheng Liu, Hang Zhao. *Fast-WAM: Do World Action Models Need Test-time Future Imagination?* arXiv:2603.16666v2, 2026. [arXiv](https://arxiv.org/abs/2603.16666) | Establishes the complementary finding that explicit future generation can be removed at inference while retaining competitive performance; motivates separating benefits of video co-training from causal use of generated futures in a particular architecture/state. |
| `zhao2026fasterwam` | Weiheng Zhao, Haoyi Jiang, Xin Shi, Liu Liu, Fan Huang, Zhizhong Su, Wei Sui, Xinggang Wang. *Faster-WAM: Efficient Inference-Time Future Conditioning for Robust World Action Models*. arXiv:2608.04404, 5 Aug 2026. [arXiv](https://arxiv.org/abs/2608.04404) | Recent efficiency/robustness evidence favoring retained future conditioning under distribution shift. |
| `qiu2026agra` | Lu Qiu, Yizhuo Li, Yi Chen, Yuying Ge, Yixiao Ge, Xihui Liu. *Making Foresight Actionable: Repurposing Representation Alignment in World Action Models*. arXiv:2606.12217, 10 Jun 2026. [arXiv](https://arxiv.org/abs/2606.12217) | Closely related content-grounding work: plausible visual futures need not yield accurate action extraction; attention analysis and perturbations motivate action-grounded representation alignment. Distinguish its training intervention and spatial robustness tests from our inference-time donor transplantation and K/V mediation. |
| `guo2024pad` | Yanjiang Guo, Yucheng Hu, Jianke Zhang, Yen-Jen Wang, Xiaoyu Chen, Chaochao Lu, Jianyu Chen. *Prediction with Action: Visual Policy Learning via Joint Denoising Process*. NeurIPS 2024, vol. 37, pp. 112386--112410. DOI 10.52202/079017-3570. [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cbe25fa0e7c7084049276888a09acc8d-Abstract-Conference.html) | Early joint future-image/action denoising architecture; supports the broader WAM lineage and the ambiguity created by joint generation. |
| `hu2025vpp` | Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, Jianyu Chen. *Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations*. ICML 2025, PMLR 267:24328--24346. [PMLR](https://proceedings.mlr.press/v267/hu25g.html) | Especially relevant to the inverse-dynamics interpretation: VPP explicitly describes its policy as an implicit inverse-dynamics model conditioned on predicted future representations. |
| `li2025uva` | Shuang Li, Yihuai Gao, Dorsa Sadigh, Shuran Song. *Unified Video Action Model*. arXiv:2503.00200v3, 2025. [arXiv](https://arxiv.org/abs/2503.00200) | Joint video/action latent modeling and decoupled decoding; contextualizes WAM design choices and future-free action inference. |
| `ye2026dreamzero` | Seonghyeon Ye et al. (36 verified authors in the BibTeX). *World Action Models are Zero-shot Policies*. arXiv:2602.15922, 17 Feb 2026. [arXiv](https://arxiv.org/abs/2602.15922) | Recent WAM framing and the claim that joint video/action prediction learns physical dynamics. Cite for model-class motivation, not as mechanistic evidence about how an internal future representation is used. |
| `liu2023libero` | Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*. NeurIPS 2023 Datasets and Benchmarks, vol. 36, pp. 44776--44791. DOI 10.52202/075280-1939. [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html) | Benchmark provenance for the ten-task LIBERO-10 experiments. |
| `nasiriany2024robocasa` | Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, Yuke Zhu. *RoboCasa: Large-Scale Simulation of Household Tasks for Generalist Robots*. RSS 2024. DOI 10.15607/RSS.2024.XX.050. [RSS proceedings](https://www.roboticsproceedings.org/rss20/p050.html) | Benchmark/simulator provenance for the exploratory cross-domain replication. |
| `yang2026robolab` | Xuning Yang, Rishit Dagli, Alex Zook, Hugo Hadfield, Ankit Goyal, Stan Birchfield, Fabio Ramos, Jonathan Tremblay. *RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies*. RSS 2026. [official NVIDIA project and BibTeX](https://research.nvidia.com/labs/srl/projects/robolab/), [arXiv](https://arxiv.org/abs/2604.09860) | Benchmark and physical-simulation provenance for the Cosmos 3 experiments. |
| `pearl2001direct` | Judea Pearl. *Direct and Indirect Effects*. UAI 2001, pp. 411--420. [UCLA author-hosted proceedings copy](https://ftp.cs.ucla.edu/pub/stat_ser/R273.pdf), [UCLA publication record](https://bayes.cs.ucla.edu/csl_papers.html) | Foundational definitions of direct, indirect, and path-specific causal effects. Use cautiously: our normalized mediation loss is an intervention estimand inspired by mediation, not automatically an identified population natural indirect effect. |
| `vig2020causalmediation` | Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, Stuart Shieber. *Investigating Gender Bias in Language Models Using Causal Mediation Analysis*. NeurIPS 2020, vol. 33, pp. 12388--12401. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/hash/92650b2e92217715fe312e6fa7b90d82-Abstract.html) | Canonical neural-network mediation citation and the encoded-versus-used distinction. |
| `meng2022rome` | Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT*. NeurIPS 2022, vol. 35, pp. 17359--17372. DOI 10.52202/068431-1262. [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html) | Causal tracing/activation restoration precedent. It supports calling the K/V replacement an activation-patching intervention, while the semantic latent transplant is more accurately a representation transplantation. |
| `goldowskydill2023pathpatching` | Nicholas Goldowsky-Dill, Chris MacLeod, Lucas Sato, Aryaman Arora. *Localizing Model Behavior with Path Patching*. arXiv:2304.05969v2, 2023. [arXiv](https://arxiv.org/abs/2304.05969) | Formal precedent for localizing behavior to hypothesized computational paths. Do not call a whole future-K/V patch a complete circuit identification. |
| `zhang2024activationpatching` | Fred Zhang, Neel Nanda. *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*. ICLR 2024. [OpenReview](https://openreview.net/forum?id=HF17y6u9BC), [arXiv](https://arxiv.org/abs/2309.16042) | Methodological support for metric choice, clean/corrupted pairing, exact no-op controls, and held-out validation. The previous local link title omitted the subtitle “Metrics and Methods”; the bibliography restores the full title. |
| `vaidyanathan2026multiplemediators` | Sankaran Vaidyanathan, David Arbour, Aaron Mueller, Scott Niekum, David Jensen. *The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching*. arXiv:2606.27510, 25 Jun 2026. [arXiv](https://arxiv.org/abs/2606.27510) | Limits additive or componentwise mediation claims: activation-patching effects can include interactions with unpatched components. Cite in limitations when interpreting layerwise or group patching. |
| `wu2024pyvene` | Zhengxuan Wu, Atticus Geiger, Aryaman Arora, Jing Huang, Zheng Wang, Noah D. Goodman, Christopher D. Manning, Christopher Potts. *pyvene: A Library for Understanding and Improving PyTorch Models via Interventions*. NAACL 2024 System Demonstrations, pp. 158--165. DOI 10.18653/v1/2024.naacl-demo.16. [ACL Anthology](https://aclanthology.org/2024.naacl-demo.16/) | General software/method precedent for interventions on internal PyTorch states. Cite only if the manuscript discusses intervention tooling or implementation semantics; the current custom denoising hooks do not depend on pyvene. |

## Discrepancies and cautions

1. **RoboCasa title.** arXiv:2406.02523 uses *Large-Scale Simulation of
   Everyday Tasks for Generalist Robots*. The final RSS proceedings record uses
   *Large-Scale Simulation of Household Tasks for Generalist Robots*. The
   bibliography cites the final proceedings title and DOI.
2. **Cosmos 3 authorship.** The arXiv record exposes a 294-person contributor
   list, while the official Cosmos 3 project explicitly says “Please cite as
   NVIDIA et al.” and supplies `author = {{NVIDIA}}`. The bibliography follows
   the official requested citation.
3. **Activation-patching title and venue.** The complete title is *Towards Best
   Practices of Activation Patching in Language Models: Metrics and Methods*;
   it is an ICLR 2024 paper. A NeurIPS 2023 ATTRIB workshop PDF also exists, but
   the bibliography cites the final ICLR record.
4. **RIFT scope.** RIFT reports physical trajectory deviation and task success,
   not success alone. Its interventions establish dependence and positional
   sensitivity. The clean novelty claim is that RIFT does not associate a
   coherent alternative future with a signed alternative action/endpoint and
   test donor-directed semantic steering.
5. **Mediation terminology.** Token-count-preserving K/V replacement is
   activation patching and can support a bounded pathway-mediation claim. It
   does not, by itself, identify a classical population natural indirect effect
   or a complete mechanistic circuit. Interaction caveats from
   `vaidyanathan2026multiplemediators` apply.
6. **Preprints versus venues.** RIFT, Cosmos Policy, Fast-WAM, Faster-WAM,
   AGRA, UVA, DreamZero, Path Patching, and The Curse of Multiple Mediators are
   represented as arXiv preprints unless a final venue was verified above. Do
   not add conference names based on search snippets, author CVs, or
   submissions under review.

## Suggested citation clusters

- **Joint prediction does not establish functional use:**
  `guo2024pad,hu2025vpp,li2025uva,kim2026cosmospolicy,ye2026dreamzero`.
- **Whether inference-time futures matter:**
  `yuan2026fastwam,zhao2026fasterwam,zhang2026rift,qiu2026agra`.
- **Closest-gap sentence:** cite `zhang2026rift` immediately after the factual
  description of RIFT; cite the present paper's method, rather than another
  source, for reachable-donor semantic steering.
- **Mechanistic intervention methodology:**
  `pearl2001direct,vig2020causalmediation,meng2022rome,goldowskydill2023pathpatching,zhang2024activationpatching,vaidyanathan2026multiplemediators`.
- **Benchmarks:** `liu2023libero,nasiriany2024robocasa,yang2026robolab`.

