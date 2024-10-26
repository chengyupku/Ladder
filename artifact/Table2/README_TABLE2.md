# Reproduce the results of Table2

The Table2 is about the compilation time of AMOS, TensorIR Welder and Ladder. The compilation time is measured by the time of the compilation.

The Table2 is generated by the following command:

```bash
python3 run_all.py --reproduce --force_tune_welder --force_tune_ladder # this may take hours to run, if you want to use paper results, you can remove the --reproduce flag, if you want to use the ladder precompiled models, you can remove the --force_tune_ladder flags.
```

The `run_all.py` script has the following options:

- `--reproduce`: bool, whether to reproduce the results, otherwise it will use our logges paper result, default value is `False`.
- `--force_tune_welder`: bool, whether to force tune the op with Welder, otherwise use the checkpoints if available, default value is `False`.
- `--force_tune_ladder`: bool, whether to force tune the op with Ladder, otherwise use the checkpoints if available, default value is `False`.
- `--force_tune_amos`: bool, whether to force tune the op with AMOS, otherwise use the checkpoints if available, default value is `False`.
- `--force_tune_tensorir`: bool, whether to force tune the op with TensorIR, otherwise use the checkpoints if available, default value is `False`.

+-----------------------------------------------------+
|          Transposed Compilation Time Table          |
+-----------------+------+----------+--------+--------+
|     Library     | AMOS | TensorIR | Welder | LADDER |
+-----------------+------+----------+--------+--------+
|    ResNet(1)    | 3852 |   156    |   11   |   31   |
|   ResNet(128)   | 3328 |   128    |   13   |   17   |
|  ShuffleNet(1)  | 2191 |   836    |   18   |   44   |
| ShuffleNet(128) | 3121 |   400    |   12   |   29   |
+-----------------+------+----------+--------+--------+

## Notes for reproducing the results

- As shown in Table2, the ML Compiler AMOS and TensorIR takes too much time to tune a whole end2end model, so we provide the tuned logs and trace files in the `$CHECKPOINTS/Table2/` directory. The `run_all.py` script will use the logs and trace files to generate the results. If you want to reproduce the results, you can set the `--force_tune_amos/--force_tune_tensorir` option to `True` to force tune the model with AMOS and TensorIR. (This may take days to finish the tuning process.)
