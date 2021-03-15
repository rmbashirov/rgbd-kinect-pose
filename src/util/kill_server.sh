#!/usr/bin/env bash

ps aux | grep -i server.py | awk '{print $2}' | xargs sudo kill -9
