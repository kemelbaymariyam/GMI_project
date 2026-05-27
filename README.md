# GMI Project

This repository contains an in-progress deep learning project for reconstructing GMI-based remote sensing data.

The goal of this project is to explore whether a **U-Net-based model** can learn to reconstruct or fill missing information in GMI-derived remote sensing data. The broader motivation is to study how deep learning can be applied to improve the usability and coverage of microwave satellite observations.

GMI is a passive microwave sensor onboard the **Global Precipitation Measurement (GPM)** mission. Its observations are useful for studying atmospheric and surface conditions, but gridded products may contain missing regions or incomplete coverage. This project investigates whether those missing patterns can be reconstructed using a deep learning approach.
The implementation is being developed based on the general methodology described in a related research paper(https://doi.org/10.1016/j.jag.2024.104029), while adapting the preprocessing and training pipeline to the current dataset.

At the moment, this repository includes code for:

- preprocessing and patch creation
- dataset loading
- U-Net training
- validation and inference experiments
- testing different strategies for handling large-scale patch data

## Status

This project is currently **in progress**.

The pipeline is still being refined, especially in terms of data storage, data loading efficiency, and practical training setup. The current version should be viewed as a research prototype under active development.
