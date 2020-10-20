# Data

The data for use in this notebook. Some external data is used where the file size is too large for GitHub. These are described [here](#external-data).

## Naming convention

Filenames are named roughly according to the following

### Temperature scales

- `SDSS` - SDSS *griz* photmetric temperature scale of Pinsonneault et al. (2012) as used in Serenelli et al. (2017)
- `ASPC` - ASPCAP spectroscopic temperature scale from SDSS DR13 as used in Serenelli et al. (2017)
- `DR14` - ASPCAP spectroscopic temperature scale from SDSS DR14
- `DR16` - ASPCAP spectroscopic temperature scale from SDSS DR16

### Spectroscopy

Metallicities used in this work are either from the ASPCAP pipeline of SDSS DR13 when using the Serenelli et al. (2017) temperatures scales, or from their respective data releases when using the SDSS `DR14` and `DR16` temperature scales.

### Targets

The main sample is that of Serenelli et al. (2017) and is refered to as `s17`. Filenames with this prefix imply the KIC IDs correspond to that of this sample. The Berger et al. (2018) sample, `b18`, was initially used for distances, but this is now depreciated in favour of using `isoclassify` with temperature scales consistent with the above.

## External Data

External data used are as follows.

### *Gaia*-*Kepler* cross-match

A 4" search radius cross-match of the KIC with *Gaia* DR2 courtesy of [GAIA-KEPLER.FUN](https://gaia-kepler.fun/). To download, click [here](https://www.dropbox.com/s/v070hmvhm2ezjax/kepler_dr2_4arcsec.fits) or run the following command,

```shell
$ wget https://www.dropbox.com/s/v070hmvhm2ezjax/kepler_dr2_4arcsec.fits
```

### Training Data

The training data will soon be available to download via Dropbox or Zenodo.
