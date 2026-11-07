# Entregas

- `submission_live_v4.csv.gz` — entrega LIVE recomendada (validada con el validador oficial: 203.083/203.083 líneas). Subir tal cual al leaderboard.
- `submission_live_v4_audit_report.md` — informe de auditoría de esa ejecución (reglas usadas, excepciones, categorías, flags).

Puntuación de referencia (scorer oficial, sin fugas): holdout TRAIN 2025-01..04 = 92.41; 2024-11..12 = 93.05; 2025-05..06 = 91.38.
Regenerar: `python3 main.py --phase live --out outputs/submission_live_v4.csv.gz` y validar con `python3 evaluation/validate.py ... --phase live`.
