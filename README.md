# TimeSeriesClassification
a collection of notebooks on time series classification projects as shown in YouTube channel @TimelyTimeSeries

(note: this repo is WIP because I'm thinking of moving all of my classification projects to this repo)

The models that I've tried so far:
- <ins>Random Convolutional Kernel for Time Series Classification (ROCKET)</ins>. This model generates thousands of random kernels, and each of these captures a feature in the input data.
  - Kaggle Tabular Playground April 2022 [notebook](https://github.com/stephanielees/TimeSeriesClassification/blob/main/tps-apr2022-rocket.ipynb) [video](https://youtu.be/0c0YNWo9Xyg)
- <ins>MiniROCKET</ins>. This model is the extension of ROCKET, so its algorithm is really similar to ROCKET, except that they change some parameter values and use some tricks to allow a more efficient computation compared to ROCKET. The biggest advantage of this model is that it can be used for a varying length of time series.
  - Asphalt classification [notebook](https://github.com/stephanielees/TimeSeriesClassification/blob/main/asphalt_classification.py) [video](https://youtu.be/D6TrmOzaVIY)
- <ins>Learning Time-series Shapelets (LTS)</ins>. This classifier is a shapelet-based classification algorithm. A shapelet is a distinctive subsequence that could tell us the category belongs to a time series.
  - Kaggle Tabular Playground April 2022 [notebook](https://github.com/stephanielees/TimeSeriesClassification/blob/main/shapelet-tslearn.ipynb) [video](https://youtu.be/u69v5gm_zBk)
- <ins>Multivariate LSTM - Fully Convolutional Network (MLSTMFCN)</ins>. This model combines Long Short Term Memory (LSTM) network with a Fully Convolutional Network. The "M" in the beginning is for Multivariate; for univariate dataset, we can use the LSTMFCN model. I applied this model on a Human Activity Recognition Data (HAR) and a US Companies Bankruptcy dataset.
  - Human Activity Recognition [notebook](https://github.com/stephanielees/TimeSeriesClassification/blob/main/UCI_HAR_MLSTMFCN_with_original_code.ipynb) [video](https://youtu.be/-rxKkserYvU)
  - US Company Bankruptcy [notebook](https://github.com/stephanielees/TimeSeriesClassification/blob/main/time-series-classification-for-business.ipynb) [video](https://youtu.be/OCQZJGItxaU
- <ins>ShapeDTW</ins>. This algorithm extends Dynamic Time Warping (DTW) by first determining the shape of the input time series before aligning pairs of sequences with the constraints usually applied in DTW. This algorithm can also be used for varying lengths of inputs, even though I haven't applied this algorithm to such datasets so far.
  - Bird sound classification [notebook](https://github.com/stephanielees/TimeSeriesClassification/blob/main/bird-song-classification-with-shapedtw.ipynb) [video](https://youtu.be/q2E-lhoJFBg) (Note: the implementation package for shapeDTW has not been updated, so I'm afraid the notebook can't be run locally)
