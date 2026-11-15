
# Data directory

Datasets are NOT stored in git (size and license). Only this file and the lists of selected
file names (`data/subsets/*.json`) are committed.

## Datasets

* **GoPro (GOPRO_Large)** -- main dataset (Nah et al., CVPR 2017).
  Source: https://seungjunnah.github.io/Datasets/gopro -- License: CC BY 4.0 (as stated on that page).
* **DIV2K** -- 25-image subset, used only as a Wiener-filter sanity check with a known blur kernel.
  Source: https://data.vision.ee.ethz.ch/cvl/DIV2K/ -- License: academic research use, as quoted in the
  Hugging Face DIV2K loader; the official page's terms were not independently checked.

## How the data is obtained and selected (both steps are reproducible)

```bash
python src/fetch_data.py --dataset div2k
python src/fetch_data.py --dataset gopro
python -m src.make_subset          # writes data/subsets/gopro_subset.json and div2k_subset.json
```

## Subset record (actual values, from the output of `python -m src.make_subset`)

| Dataset                    | Split        | Count     | Selection method                                                                                       | Seed |
| -------------------------- | ------------ | --------- | ------------------------------------------------------------------------------------------------------ | ---- |
| GoPro (`blur`folder)     | train        | 365 pairs | every 5th frame of 19 training sequences                                                               | 42   |
| GoPro (`blur`folder)     | val          | 56 pairs  | every 5th frame of 3 whole training sequences held out: GOPR0372_07_00, GOPR0374_11_01, GOPR0881_11_00 | 42   |
| GoPro (`blur`folder)     | test         | 223 pairs | every 5th frame of the 11 official test sequences                                                      | 42   |
| DIV2K (`DIV2K_valid_HR`) | sanity check | 25 images | random sample of the 100 images                                                                        | 42   |

GoPro also provides a `blur_gamma` folder (a second blurred version of the same frames); this project
uses the `blur` folder only, and the choice is recorded in `gopro_subset.json` (`"blur_dir": "blur"`).
