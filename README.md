# GMI Project

This repository contains an in-progress deep learning project for reconstructing GMI-based remote sensing data.

The main idea is to test whether a U-Net model can be trained to reconstruct or fill missing information in gridded GMI observations. The implementation is being developed based on the general methodology described in a related research paper, while adapting the preprocessing and training pipeline to the current dataset.

At the moment, this repository includes code for:

- preprocessing and patch creation
- dataset loading
- U-Net training
- validation and inference experiments
- testing different strategies for handling large-scale patch data

## Status

This project is currently **in progress**.

The pipeline is still being refined, especially in terms of data storage, data loading efficiency, and practical training setup. The current version should be viewed as a research prototype under active development.
