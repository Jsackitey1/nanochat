# nanochat

![nanochat logo](dev/nanochat.png)

> The best ChatGPT that $100 can buy.

> [!NOTE]
> This repository is a fork of the original [nanochat](https://github.com/karpathy/nanochat) by **Andrej Karpathy**. I am tweaking this version to train it specifically on **Electricity and Magnetism** concepts. The goal is to quiz the model to evaluate how well it learns this specialized domain.

This repo is a full-stack implementation of an LLM like ChatGPT in a single, clean, minimal, hackable, dependency-lite codebase. nanochat is designed to run on a single 8XH100 node via scripts like [speedrun.sh](speedrun.sh), that run the entire pipeline start to end. This includes tokenization, pretraining, finetuning, evaluation, inference, and web serving over a simple UI so that you can talk to your own LLM just like ChatGPT. nanochat will become the capstone project of the course LLM101n being developed by Eureka Labs.

## Project: E&M Tutor Micro-LLM (Explain, Solve, Check Units)

### What it should do (clear, measurable behaviors)

1. **Explain concepts** (Gauss’s law, fields vs potential, capacitance, circuits basics, magnetism, induction).
2. **Solve problems step-by-step** with correct equations.
3. **Show unit checks** and sanity checks (order of magnitude, direction of field, sign conventions).
4. **Refuse to guess** when missing info and ask for the missing variable.

### What you will train

You will train 2 models (simple but research-legit):

* **Baseline**: nanochat as-is (after speedrun).
* **E&M Tutor**: the same model, then instruction-fine-tuned on your custom E&M tutoring dataset.

Then you compare them.

---

## Your dataset (the secret sauce)

You do not need millions of examples. You need **high quality**.

### Format for each training example

A single JSONL line like this:

* **Input**: problem statement + any givens + what the student asks
* **Output**: explanation + solution steps + unit check + final answer

Example topics to include:

* Coulomb’s law, superposition
* Electric field from point charges and continuous charge (simple symmetry)
* Gauss’s law for sphere, cylinder, plane
* Potential vs field (relationship, sign)
* Capacitance, energy in capacitor
* Basic DC circuits (Ohm’s law, series/parallel, power)

Target: **200 to 800** high-quality examples.

---

## Evaluation (so it counts as research)

We use a small held-out test set (50 to 150 questions) and score:

1. **Final answer correctness**
2. **Reasoning correctness** (equations and steps make sense)
3. **Unit correctness**
4. **Hallucination rate** (makes up constants, wrong units, invents givens)

---

## Roadmap

1. **Scope v1**: Electrostatics + Gauss + Potential + Capacitors.
2. **Dataset Generation**: Create a generator template to write examples quickly.
3. **Baseline**: Train baseline (speedrun) and run evaluation prompts.
4. **Broadcast**: Fine-tune with the dataset and rerun evaluation.

## File structure

```
.
├── LICENSE
├── README.md
├── dev                     # Dev tools & scripts
├── nanochat                # Core library
│   ├── gpt.py              # The Transformer
│   ├── tokenizer.py        # Tokenizer
│   └── ...
├── scripts                 # Training & Eval scripts
│   ├── base_train.py
│   ├── chat_sft.py
│   └── ...
├── em_tutor_dataset_builder.py # [NEW] Dataset generator
├── em_tutor_train.jsonl    # [NEW] Generated training data
└── tests
```


## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
