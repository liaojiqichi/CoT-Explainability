from datasets import load_dataset

def load_reasoning_dataset(
    name="gsm8k",
    sample_size=200,
    seed=42,
):

    if name == "gsm8k":

        dataset = (
            load_dataset(
                "openai/gsm8k",
                "main",
                split="test"
            )
            .shuffle(seed=seed)
            .select(range(sample_size))
        )

        return dataset

    raise ValueError(
        f"Unknown dataset: {name}"
    )
