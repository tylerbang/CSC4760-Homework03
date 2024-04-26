#!/bin/bash

find . -type f \( -name "*.o" -o -name "*.h" -o -name "*.hpp" -o -name "*.tmp" -o -name "*.a" -o -name "*.host" \) -delete
rm -r desul