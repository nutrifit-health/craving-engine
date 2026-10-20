Work only on candidate/checkpoint.py. Implement save(memory) -> JSON string and
restore(memory_class, serialized) -> memory. Preserve complete V12 state, including
pending feedback, configuration and future learning. Use memory.export_state() and
memory_class.from_state(serialized). Do not fit or tune models.
Finish with one JSON object containing answer (concise explanation) and files
(the complete candidate/checkpoint.py contents). Do not create output/answer.md.
Once the two wrapper calls are established, return immediately; independent
correctness probes run in the OCI judge. Do not repeat inventories or run local
model-equivalence experiments.
