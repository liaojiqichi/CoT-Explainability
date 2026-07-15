"""
Prompt templates
"""

ALGEBRA_PROMPT = (
    "Solve step by step using algebra.\n"
    "Let x be the unknown variable."
)

ARITHMETIC_PROMPT = (
    "Solve step by step using arithmetic only."
)

VERBOSE_PROMPT = (
    "Think carefully.\n"
    "Explain every reasoning step."
)

CONCISE_PROMPT = (
    "Reason briefly."
)

SCRATCHPAD_PROMPT = (
    "Write intermediate calculations."
)

NATURAL_PROMPT = (
    "Reason naturally as a human."
)

SYMBOLIC_PROMPT = (
    "Use symbolic reasoning whenever possible."
)

PROGRAM_PROMPT = (
    "Solve using program-like reasoning."
)

TREE_PROMPT = (
    "Explore multiple reasoning paths before deciding."
)

PROMPT_FAMILIES = {
    "algebra": ALGEBRA_PROMPT,
    "arithmetic": ARITHMETIC_PROMPT,
    "verbose": VERBOSE_PROMPT,
    "concise": CONCISE_PROMPT,
    "scratchpad": SCRATCHPAD_PROMPT,
    "natural": NATURAL_PROMPT,
    "symbolic": SYMBOLIC_PROMPT,
    "program": PROGRAM_PROMPT,
    "tree": TREE_PROMPT,
}
