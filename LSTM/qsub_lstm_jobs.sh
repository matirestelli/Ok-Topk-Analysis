#!/bin/bash -l
# Launcher script for Polaris - submits all LSTM job variants
qsub lstm_dense_polaris.sh
qsub lstm_gaussiank_polaris.sh
qsub lstm_gtopk_polaris.sh
qsub lstm_oktopk_polaris.sh
qsub lstm_topkA_polaris.sh
qsub lstm_topkDSA_polaris.sh
