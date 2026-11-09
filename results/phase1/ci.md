| condition | adv_violation [95% CI] | false_refusal [95% CI] | recovery [95% CI] |
|---|---|---|---|
| dpo | 0.194 [0.097, 0.290] | 0.000 [0.000, 0.000] | 0.726 [0.613, 0.839] |
| opcd | 0.161 [0.065, 0.258] | 0.106 [0.062, 0.156] | 0.581 [0.452, 0.710] |
| opcd_gr | 0.113 [0.032, 0.194] | 0.131 [0.081, 0.181] | 0.452 [0.339, 0.581] |
| student_none | 0.516 [0.387, 0.629] | 0.000 [0.000, 0.000] | 0.242 [0.129, 0.355] |
| student_prompt | 0.113 [0.032, 0.194] | 0.369 [0.294, 0.438] | 0.371 [0.258, 0.500] |
| teacher_prompt | 0.081 [0.016, 0.145] | 0.019 [0.000, 0.044] | 0.435 [0.306, 0.548] |

Exact McNemar (paired, same prompts). a_only = prompts where only the first condition has the event.

| comparison | metric | a_only | b_only | p |
|---|---|---|---|---|
| opcd vs student_prompt | adv_violation | 8 | 5 | 0.5811 |
| opcd vs student_prompt | false_refusal | 1 | 43 | <0.0001 |
| opcd vs student_prompt | recovery | 15 | 2 | 0.0023 |
| opcd_gr vs student_prompt | adv_violation | 5 | 5 | 1.0 |
| opcd_gr vs student_prompt | false_refusal | 9 | 47 | <0.0001 |
| opcd_gr vs student_prompt | recovery | 10 | 5 | 0.3018 |
| dpo vs student_prompt | adv_violation | 12 | 7 | 0.3593 |
| dpo vs student_prompt | false_refusal | 0 | 59 | <0.0001 |
| dpo vs student_prompt | recovery | 26 | 4 | 0.0001 |
| opcd_gr vs opcd | adv_violation | 0 | 3 | 0.25 |
| opcd_gr vs opcd | false_refusal | 18 | 14 | 0.5966 |
| opcd_gr vs opcd | recovery | 3 | 11 | 0.0574 |
| dpo vs opcd | adv_violation | 12 | 10 | 0.8318 |
| dpo vs opcd | false_refusal | 0 | 17 | <0.0001 |
| dpo vs opcd | recovery | 16 | 7 | 0.0931 |
| opcd vs student_none | adv_violation | 3 | 25 | <0.0001 |
| opcd vs student_none | false_refusal | 17 | 0 | <0.0001 |
| opcd vs student_none | recovery | 24 | 3 | <0.0001 |
