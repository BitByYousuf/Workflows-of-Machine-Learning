configfile: "configs/config.yaml"

include: "workflow/rules/train.smk"
include: "workflow/rules/test.smk"

rule all:
    input:
        "results/metrics.txt"