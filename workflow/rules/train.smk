rule train_model:
    input:
        data=config["data"]
    output:
        model="results/model.pkl",
        test_data="results/test_data.pkl"
    script:
        "../scripts/train.py"