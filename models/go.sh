#!/usr/bin/env bash

rm csvs/markov4/patch_csvs/*.csv
rm csvs/markov4/*.csv
./markov.py
cd ..
./get_patches.py
./cli.py
