| rule | dataset | subset | n | prevalence | precision | recall | F1 | kappa | verdict |
|---|---|---|---|---|---|---|---|---|---|
| refusal_opening **(primary)** | XSTest v2 completions | all 5 models | 2250 | 0.515 | 0.987 [0.976, 0.996] | 0.444 [0.415, 0.470] | 0.612 [0.583, 0.637] | 0.430 [0.398, 0.458] | weak |
| refusal_opening | XSTest v2 completions | gpt4 | 450 | 0.489 | 1.000 [1.000, 1.000] | 0.041 [0.017, 0.066] | 0.079 [0.034, 0.123] | 0.042 [0.017, 0.068] | not supported |
| refusal_opening | XSTest v2 completions | llama2new | 450 | 0.609 | 0.988 [0.968, 1.000] | 0.584 [0.523, 0.641] | 0.734 [0.683, 0.778] | 0.514 [0.445, 0.580] | weak |
| refusal_opening | XSTest v2 completions | llama2orig | 450 | 0.776 | 0.993 [0.981, 1.000] | 0.759 [0.714, 0.804] | 0.860 [0.830, 0.890] | 0.574 [0.494, 0.646] | weak |
| refusal_opening | XSTest v2 completions | mistralguard | 450 | 0.533 | 0.962 [0.920, 1.000] | 0.317 [0.261, 0.376] | 0.476 [0.409, 0.539] | 0.289 [0.230, 0.354] | not supported |
| refusal_opening | XSTest v2 completions | mistralinstruct | 450 | 0.169 | 1.000 [1.000, 1.000] | 0.053 [0.012, 0.108] | 0.100 [0.024, 0.195] | 0.085 [0.019, 0.168] | not supported |
| refusal_anywhere | XSTest v2 completions | all 5 models | 2250 | 0.515 | 0.985 [0.974, 0.994] | 0.448 [0.419, 0.473] | 0.616 [0.587, 0.640] | 0.433 [0.401, 0.461] | weak |
| refusal_anywhere | XSTest v2 completions | gpt4 | 450 | 0.489 | 1.000 [1.000, 1.000] | 0.045 [0.021, 0.073] | 0.087 [0.040, 0.136] | 0.046 [0.021, 0.076] | not supported |
| refusal_anywhere | XSTest v2 completions | llama2new | 450 | 0.609 | 0.988 [0.968, 1.000] | 0.588 [0.527, 0.643] | 0.737 [0.688, 0.780] | 0.518 [0.450, 0.583] | weak |
| refusal_anywhere | XSTest v2 completions | llama2orig | 450 | 0.776 | 0.989 [0.975, 1.000] | 0.765 [0.720, 0.811] | 0.863 [0.833, 0.891] | 0.575 [0.496, 0.647] | weak |
| refusal_anywhere | XSTest v2 completions | mistralguard | 450 | 0.533 | 0.963 [0.920, 1.000] | 0.321 [0.265, 0.380] | 0.481 [0.414, 0.545] | 0.293 [0.234, 0.358] | not supported |
| refusal_anywhere | XSTest v2 completions | mistralinstruct | 450 | 0.169 | 1.000 [1.000, 1.000] | 0.053 [0.012, 0.108] | 0.100 [0.024, 0.195] | 0.085 [0.019, 0.168] | not supported |
| recovery_marker | XSTest v2 completions | all 5 models | 2250 | 0.515 | 0.876 [0.857, 0.896] | 0.840 [0.820, 0.861] | 0.858 [0.842, 0.873] | 0.713 [0.685, 0.742] | supported |
| recovery_marker | XSTest v2 completions | gpt4 | 450 | 0.489 | 0.928 [0.896, 0.959] | 0.936 [0.903, 0.966] | 0.932 [0.908, 0.954] | 0.867 [0.822, 0.907] | supported |
| recovery_marker | XSTest v2 completions | llama2new | 450 | 0.609 | 0.910 [0.877, 0.943] | 0.920 [0.885, 0.950] | 0.915 [0.890, 0.937] | 0.780 [0.723, 0.832] | supported |
| recovery_marker | XSTest v2 completions | llama2orig | 450 | 0.776 | 0.951 [0.926, 0.974] | 0.954 [0.931, 0.974] | 0.953 [0.936, 0.969] | 0.789 [0.720, 0.854] | supported |
| recovery_marker | XSTest v2 completions | mistralguard | 450 | 0.533 | 0.816 [0.762, 0.872] | 0.667 [0.608, 0.730] | 0.734 [0.686, 0.783] | 0.489 [0.409, 0.572] | weak |
| recovery_marker | XSTest v2 completions | mistralinstruct | 450 | 0.169 | 0.343 [0.232, 0.459] | 0.303 [0.206, 0.400] | 0.322 [0.227, 0.414] | 0.194 [0.090, 0.306] | not supported |
| refusal_opening | XSTest v2 completions | positive = full refusal only | 2250 | 0.437 | 0.875 [0.844, 0.901] | 0.464 [0.433, 0.494] | 0.606 [0.577, 0.633] | 0.435 [0.402, 0.467] | weak |
| refusal_opening | JailbreakBench judge_comparison | harmful = no refusal | 300 | 0.367 | 0.412 [0.355, 0.466] | 0.982 [0.955, 1.000] | 0.581 [0.519, 0.632] | 0.133 [0.082, 0.191] | not supported |
| refusal_anywhere | JailbreakBench judge_comparison | harmful = no refusal | 300 | 0.367 | 0.415 [0.357, 0.470] | 0.982 [0.955, 1.000] | 0.584 [0.522, 0.636] | 0.141 [0.090, 0.200] | not supported |
| abuse_lexicon | BeaverTails 30k_test | hate / offensive / discrimination rows | 438 | 1.000 | 1.000 [1.000, 1.000] | 0.144 [0.114, 0.176] | 0.252 [0.205, 0.299] | 0.000 [0.000, 0.000] | not supported |
| abuse_lexicon | BeaverTails 30k_test | all rows | 3021 | 0.574 | 0.830 [0.760, 0.897] | 0.048 [0.038, 0.058] | 0.091 [0.073, 0.109] | 0.030 [0.020, 0.041] | not supported |
| abuse_lexicon **(primary)** | BeaverTails 30k_test | label = hate_speech,offensive_language flag | 3021 | 0.100 | 0.600 [0.510, 0.695] | 0.199 [0.153, 0.244] | 0.298 [0.239, 0.357] | 0.262 [0.203, 0.320] | not supported |
| abuse_lexicon | ToxicChat 0124 test | user_input vs toxicity | 5083 | 0.071 | 0.565 [0.368, 0.773] | 0.036 [0.020, 0.057] | 0.068 [0.038, 0.106] | 0.059 [0.031, 0.096] | not supported |
| qwen3guard_refusal | XSTest v2 completions | all 5 models | 2250 | 0.515 | 0.944 [0.931, 0.959] | 0.894 [0.875, 0.911] | 0.918 [0.906, 0.930] | 0.837 [0.814, 0.859] | supported |
| qwen3guard_unsafe | JailbreakBench judge_comparison | harmful = Unsafe/Controversial | 300 | 0.367 | 0.747 [0.673, 0.814] | 0.991 [0.971, 1.000] | 0.852 [0.802, 0.896] | 0.745 [0.671, 0.818] | supported |
