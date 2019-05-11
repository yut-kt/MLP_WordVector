#!/usr/bin/env bash

echo 'start: install'
git clone https://github.com/yut-kt/WordVector.git
mv WordVector/preprocessing .
rm -rf WordVector

echo 'success install'