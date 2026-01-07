#!/bin/bash
# run this with a fresh clone of the repository

mk_a_dir() {
    if [[ ! -d "$1" ]]; then
    mkdir "$1"
    fi
}

mk_a_dir "./cache"
mk_a_dir "./data"
mk_a_dir "./media"
mk_a_dir "./papers"
mk_a_dir "./results"
mk_a_dir "./tmp"

cp -R ./eg_data/* ./data
