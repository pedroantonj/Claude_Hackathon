# bankrec — conciliación bancaria multiagente (Hackathon Talky × Anthropic)

Sistema híbrido: reglas contables deterministas + matching (claves de ficheros, nombre truncado, subset-sum, RapidFuzz)
+ detección de anomalías (devoluciones, duplicados) + clasificación de excepciones + un agente LLM **opcional** que sólo
revisa los casos ambiguos (nunca por transacción).

## Requisitos

- Python ≥ 3.10, pandas, numpy, openpyxl (ya instalados) y `rapidfuzz` (instalado con `pip3 install --user rapidfuzz`).
- PDFs (facturas, avisos de remesa) se leen con la librería estándar (ReportLab: ASCII85 + Flate), sin dependencias.
- Datos en `../data/` (`train/`, `live/`, `final/` extraídos de los `.tar.gz`; `masters`, `sample`, `starter` son
  enlaces a `bankrec-docs-and-masters/`). Ruta configurable con `BANKREC_DATA`.
- RAM: ~7 GB por ejecución (LIVE o ventana TRAIN de 2-6 meses).

## Puesta en marcha (para el equipo)

Los datos del hackathon **no están en el repo**. Estructura esperada (carpeta `data/` al lado del repo):

```
Claude Hackathon/
├── bankrec/                         <- este repo
├── bankrec-docs-and-masters/        <- descomprimido de bankrec-docs-and-masters.zip
└── data/
    ├── train/   live/               <- tar -xzf bankrec-train-lite.tar.gz -C data ; tar -xzf bankrec-live.tar.gz -C data
    ├── masters -> ../bankrec-docs-and-masters/masters
    ├── sample  -> ../bankrec-docs-and-masters/sample
    └── starter -> ../bankrec-docs-and-masters/starter
```

```bash
mkdir -p data && tar -xzf bankrec-train-lite.tar.gz -C data && tar -xzf bankrec-live.tar.gz -C data
ln -sfn ../bankrec-docs-and-masters/masters data/masters
ln -sfn ../bankrec-docs-and-masters/sample  data/sample
ln -sfn ../bankrec-docs-and-masters/starter data/starter
pip3 install -r bankrec/requirements.txt
cd bankrec && python3 main.py --phase sample --out outputs/sample.csv.gz   # prueba rápida (~10 s)
```

Si los datos están en otro sitio: `export BANKREC_DATA=/ruta/a/data`.

## Ejecutar

```bash
cd bankrec
python3 main.py --phase live  --out outputs/submission_live.csv.gz          # entrega LIVE (~3-5 min)
python3 evaluation/validate.py outputs/submission_live.csv.gz --phase live  # validación offline = reglas del servidor
python3 main.py --phase final --out outputs/submission_final.csv.gz         # FINAL (cuando se publiquen los datos)

# Desarrollo / evaluación sin fugas (ventana de TRAIN; saldos abiertos reconstruidos con golden ANTERIOR a la ventana)
python3 tools/dev.py --from 2025-05-01 --to 2025-06-30 --dump /tmp/err.csv.gz
python3 evaluation/evaluate.py outputs/X.csv.gz --phase train --from 2025-01 --to 2025-06
python3 tests/run_tests.py                                                   # tests (también compatibles con pytest)

# Agente LLM opcional sobre la cola de revisión (por defecto dry-run, sin coste)
python3 agents/llm_reviewer.py outputs/submission_live_review_queue.csv [--enable --max-cases 200]
```

Cada ejecución produce: la entrega (`*.csv.gz`), la traza de auditoría por línea (`*_audit_trail.csv.gz`: regla,
confianza, nº de partidas, nota), la cola de revisión (`*_review_queue.csv`, confianza < 0.75) y el informe
(`*_audit_report.md`).

## Arquitectura

```
main.py                     orquesta: contexto -> motor -> CSV + auditoría (PYTHONHASHSEED=0, determinista)
data_loader/                lectura (bank, items, batches, PDFs stdlib) + contexto por fase / ventana dev sin fugas
matching/engine.py          pasada cronológica única con saldos abiertos vivos (State)
matching/files.py           remesas de pago (e2e/run), SEPA DD, cheques, liquidaciones de tarjeta
matching/remittance.py      avisos de remesa (PDF/CSV/XLSX/TXT)
matching/party.py           pagador (prefijo truncado, IBAN, padre/hijo) + importe: refs, exacto, FIFO, subset-sum
matching/learned.py         alias aprendidos SOLO del golden histórico (antes de la ventana/fase)
exceptions/short_payments.py  FX, pronto pago, retenciones, comisiones, parciales (POLICIES §3)
exceptions/categories.py    líneas sin partida (POLICIES §4) + post-proceso (pares BANK_REVERSAL)
anomaly_detection/          devoluciones (RETURNED), pagos duplicados, flags de auditoría
agents/reporting.py         traza, cola de revisión, informe; agents/llm_reviewer.py agente LLM opcional
evaluation/                 evaluate.py (scorer oficial importado) + validate.py
tests/                      tests automáticos
```

Orden por línea (gana la primera regla): categoría de tesorería segura → devolución → ficheros (remesas de pago, DD,
cheques, tarjeta) → aviso de remesa → pagador + importe → duplicado → categoría por defecto (cobro sin evidencia =
`UNIDENTIFIED`, nunca se fuerza un match sólo por importe).

## Equipo multiagente (cómo se construyó)

| Agente | Responsable de |
|---|---|
| 1 Data Engineer | inventario de datos, loaders, PDFs stdlib, contexto LIVE/FINAL, rendimiento |
| 2 Matching Engineer | motor de matching (ficheros, avisos, pagador+importe, subset-sum, fuzzy) |
| 3 Fraud & Anomaly | devoluciones, duplicados y reembolsos, pares de error bancario, flags |
| 4 Accounting & Exceptions | categorías sin partida, descuentos, retenciones, FX, comisiones, parciales |
| 5 QA & Evaluation | scorer, harness sin fugas, validador, tests, revisión independiente |
| Coordinador | arquitectura, integración, propiedad de ficheros, ejecuciones serializadas (lock) |
