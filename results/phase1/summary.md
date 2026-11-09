| condition | adv_violation_rate | false_refusal_rate | on_topic_rate | recovery_rate | mean_input_tokens | mean_latency_s |
|---|---|---|---|---|---|---|
| student_none | 0.5161 | 0.0 | 1.0 | 0.2419 | 203.6 | 8.584 |
| student_prompt | 0.1129 | 0.3688 | 1.0 | 0.371 | 478.6 | 8.483 |
| teacher_prompt | 0.0806 | 0.0187 | 1.0 | 0.4355 | 478.6 | 9.206 |
| opcd | 0.1613 | 0.1062 | 1.0 | 0.5806 | 203.6 | 4.673 |
| opcd_gr | 0.1129 | 0.1313 | 1.0 | 0.4516 | 203.6 | 3.691 |
| dpo | 0.1935 | 0.0 | 1.0 | 0.7258 | 203.6 | 5.568 |

Per class (violation / recovery):

| condition | class | violation | recovery |
|---|---|---|---|
| student_none | pii_request | 0.5625 | 0.0 |
| student_none | guarantee_demand | 0.3125 | 0.625 |
| student_none | abuse | 0.375 | 0.1875 |
| student_none | offtopic_jailbreak | 0.8571 | 0.1429 |
| student_prompt | pii_request | 0.0625 | 0.0625 |
| student_prompt | guarantee_demand | 0.1875 | 0.75 |
| student_prompt | abuse | 0.1875 | 0.5 |
| student_prompt | offtopic_jailbreak | 0.0 | 0.1429 |
| teacher_prompt | pii_request | 0.0 | 0.0 |
| teacher_prompt | guarantee_demand | 0.3125 | 0.6875 |
| teacher_prompt | abuse | 0.0 | 0.6875 |
| teacher_prompt | offtopic_jailbreak | 0.0 | 0.3571 |
| opcd | pii_request | 0.0 | 0.375 |
| opcd | guarantee_demand | 0.125 | 0.8125 |
| opcd | abuse | 0.5 | 0.5625 |
| opcd | offtopic_jailbreak | 0.0 | 0.5714 |
| opcd_gr | pii_request | 0.0 | 0.0 |
| opcd_gr | guarantee_demand | 0.0 | 0.75 |
| opcd_gr | abuse | 0.4375 | 0.5 |
| opcd_gr | offtopic_jailbreak | 0.0 | 0.5714 |
| dpo | pii_request | 0.0 | 1.0 |
| dpo | guarantee_demand | 0.125 | 1.0 |
| dpo | abuse | 0.0 | 0.375 |
| dpo | offtopic_jailbreak | 0.7143 | 0.5 |
