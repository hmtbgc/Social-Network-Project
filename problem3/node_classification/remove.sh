#!/bin/bash
name=$1
rm -rf log/${name}
rm -rf tensorboard_log/${name}
rm -rf model_pt/${name}