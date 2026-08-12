# caiman_tools

CLI tools for batch-processing calcium imaging movies with [CaImAn](https://github.com/flatironinstitute/CaImAn).

## Installation

1. Set up a conda environment with CaImAn:

   ```bash
   conda create -n caiman caiman ipykernel python=3.12
   conda activate caiman
   ```

2. Install `caiman_tools` into that environment, straight from GitHub:

   ```bash
   python -m pip install "git+ssh://git@github.com/sommerc/caiman_tools.git"
   ```

   To upgrade to the latest version later, add `--upgrade`:

   ```bash
   python -m pip install --upgrade "git+ssh://git@github.com/sommerc/caiman_tools.git"
   ```

This installs the `caiman-cnmfe` command-line tool into the `caiman` environment.

## Usage: `caiman-cnmfe`

`caiman-cnmfe` runs the full CNMF-E pipeline (motion correction -> CNMF-E source extraction ->
component evaluation -> dF/F) non-interactively over one or more tif movies. For each `movie.tif`
it writes `movie_caiman.hdf5` next to the input.

Basic usage:

```bash
conda activate caiman
caiman-cnmfe movie.tif
```

Batch usage (on all `*.tif` below `/root/path/to/movies`):

```bash
find /root/path/to/movies -name "*.tif" -print0 | xargs -0 caiman-cnmfe --n-processes 32
```

### Options

| Flag | Description |
| --- | --- |
| `--params PARAMS.json` | JSON file overriding the default CNMF-E parameters (only the keys you provide are overridden — see below). |
| `--n-processes N` | Number of worker processes for the shared multiprocessing cluster. Defaults to `pipeline.n_processes` in `--params`, or CaImAn's own auto-detected default. |
| `--overwrite` | Reprocess and overwrite outputs that already exist (by default, files with an existing `_caiman.hdf5` output are skipped). |
| `--cleanup-memmap` | Delete intermediate `.mmap` files after a successful save. |
| `--skip-view` | Don't auto-generate the `_caiman_view.html` interactive component viewer after saving. |
hidden). `caiman_tools`' own progress/summary output is always shown. |

### Overriding parameters

Download the [default params.json](https://raw.githubusercontent.com/sommerc/caiman_tools/main/src/caiman_tools/default_cnmfe_params.json)
as a starting point. To override a subset, pass a JSON file with only the keys you want to change,
e.g. to adjust the CNMF-E initialization thresholds:

```json
{
  "cnmf_params": {
    "min_corr": 0.9,
    "min_pnr": 15
  }
}
```

```bash
caiman-cnmfe movie.tif --params my_params.json
```
