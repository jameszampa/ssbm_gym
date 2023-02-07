#!/bin/bash

waitress-serve --listen=0.0.0.0:$1 --asyncore-use-poll --threads 64 melee_server_main:app