# Brute-Force Attack Detection: Meta-Analysis (Supplementary File S1)

Data and analysis code for:

**Machine Learning and Deep Learning for Brute-Force Attack Detection: A Systematic Review and Three-Level Meta-Analysis of Reported Performance**

`S1_brute_force_meta_analysis.py` is a single self-contained file that holds:

1. **The extracted data:** 21 model-level accuracy estimates from 8 studies, and 14 recall estimates from 7 studies.
2. **The complete analysis** that reproduces Table 6 and the values reported in the text:
   - logit transformation of accuracy with binomial sampling variance
   - three-level random-effects model (REML), which is the primary analysis
   - model-level DerSimonian–Laird and REML models, with the HKSJ adjustment
   - study-level model, leave-one-study-out, and exclusion of the LSTM outlier
   - subgroup analysis by model family (classical ML, DL, hybrid DL)
   - Egger's regression test for small-study effects
   - three-level meta-analysis of recall

## Requirements

Python 3.9 or later, plus numpy and scipy:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python S1_brute_force_meta_analysis.py                  # print the results report
python S1_brute_force_meta_analysis.py --csv data.csv   # also write the extracted data (with logit, variance, SE)
python S1_brute_force_meta_analysis.py --json res.json  # also write all results as JSON
```

## Included studies

| Key | Reference |
|---|---|
| fakrullah | Fakrullah et al., Int. J. Innov. Comput. 13(1), 2023. doi:10.11113/ijic.v13n1.386 |
| alotibi | Alotibi & Alshammari, IJACSA 14(6), 2023. doi:10.14569/IJACSA.2023.0140612 |
| akbar | Akbar et al., Internet Things Artif. Intell. J. 3(3), 2023. doi:10.31763/iota.v3i3.630 |
| agghey | Agghey et al., Symmetry 13(11), 2021. doi:10.3390/sym13112192 |
| odeh | Odeh & Abu Taleb, Appl. Sci. 13(21), 2023. doi:10.3390/app132111985 |
| bello | Bello Suleiman et al., IJISRT, 2024. doi:10.38124/ijisrt/IJISRT24JUL160 |
| wanjau | Wanjau et al., IJCATR 10(01), 2021. doi:10.7753/IJCATR1001.1008 |
| gouda | Gouda et al., Neural Comput. Appl. 36(6), 2024. doi:10.1007/s00521-023-09309-y |

The review also included Zhang et al. (IEEE Access, 2020), but that study reported no detection accuracy.
