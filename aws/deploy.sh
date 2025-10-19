#!/usr/bin/env bash

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source $script_dir/aws.env

project_dir_path="$server_home_dir/$project_dir_name"

target="x86_64-unknown-linux-gnu"
build="debug"
binary="phile"

file="$script_dir/../target/$target/$build/$binary"

echo
echo "Deploying"
echo
rsync -a -P $file $ssh_connection:$project_dir_path
