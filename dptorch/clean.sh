#!/bin/bash
rm "${PWD##*/}".*.out "${PWD##*/}".*.err
rm -r runs/* postprocess/*