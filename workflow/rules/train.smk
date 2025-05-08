rule train_model:
    input:
        data=config["data"]
    output:
        model="results/model.pkl",
        test_data="results/test_data.pkl"
    log:
        "log/train_model.log"
    benchmark:
        "benchmarks/train_model.txt"
    script:
        "../scripts/train.py"